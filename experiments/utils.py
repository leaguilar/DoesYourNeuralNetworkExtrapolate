"""
Shared utilities for feature engineering / OOD generalization experiments.
NeurIPS 2026 — "Feature Engineering as Inductive Bias"
"""

import json
import os
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_timing() -> float:
    """Reset GPU peak-memory counter and return a wall-clock t0 (time.time())."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.time()


def collect_meta(t0: float) -> dict:
    """Collect per-run timing/device metadata as a flat dict that can be merged
    into an experiment's result JSON. Picked up by `resources_tables.py` via
    the `wall_time_s` / `gpu_name` / `gpu_mem_peak_gb` keys."""
    meta = {
        "wall_time_s":     float(time.time() - t0),
        "device":          str(DEVICE),
        "gpu_name":        None,
        "gpu_mem_peak_gb": None,
    }
    if torch.cuda.is_available():
        try:
            meta["gpu_name"]        = torch.cuda.get_device_name(0)
            meta["gpu_mem_peak_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            pass
    return meta


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


_SRC_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def save_result(result: dict, output_dir: str = None):
    """Save experiment result as JSON.

    Default path: results/exp{id}/exp{id}__seed{seed}__{ts}.json
    Pass output_dir to override the per-experiment subdirectory.
    """
    exp_id = result["experiment_id"].replace(".", "_")
    if output_dir is None:
        output_dir = os.path.join(_SRC_RESULTS_DIR, f"exp{exp_id}")
    os.makedirs(output_dir, exist_ok=True)
    seed = result.get("seed", 0)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = f"exp{exp_id}__seed{seed}__{ts}.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {path}")


def ood_relative_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean relative percent error (Vasiliauskaite et al., Eq. 8)."""
    return float(np.abs(y_pred - y_true).mean() / (np.abs(y_true).mean() + 1e-12) * 100.0)


def spectral_complexity(X_rep: np.ndarray, y: np.ndarray) -> float:
    """Fraction of variance in y NOT explained by a linear model of X_rep.

    0.0 = perfectly linear in this representation (low complexity, good for OOD).
    1.0 = no linear structure (high complexity, bad for OOD).
    """
    X = np.hstack([X_rep.reshape(len(X_rep), -1), np.ones((len(X_rep), 1))])
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(ss_res / ss_tot)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, layers: int = 5, out_dim: int = 1):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def train_mlp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = False,
    device: torch.device = None,
) -> nn.Module:
    if device is None:
        device = DEVICE
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    model.train()
    n = X.shape[0]
    # Full-batch when data fits comfortably on device — avoids DataLoader overhead.
    # On CUDA, full-batch is the fast path because mini-batch with small batches
    # underutilizes the GPU (high overhead-to-compute ratio).
    if device.type == "cuda" or n <= 10000:
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 500 == 0:
                print(f"    epoch {epoch+1}/{epochs}  loss={loss.item():.4e}")
    else:
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            if verbose and (epoch + 1) % 500 == 0:
                print(f"    epoch {epoch+1}/{epochs}  loss={loss.item():.4e}")
    return model


def evaluate_mlp(model: nn.Module, X_test: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_test, dtype=torch.float32, device=device)
        return model(X).squeeze(-1).cpu().numpy()


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit OLS with bias; return weight vector (including bias as last element)."""
    A = np.hstack([X, np.ones((len(X), 1))])
    w, _, _, _ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    return w


def predict_ols(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    A = np.hstack([X, np.ones((len(X), 1))])
    return (A @ w).ravel()


def run_multiseed(fn, seeds=(0, 1, 2)) -> list[dict]:
    """Run fn(seed) for each seed and return list of result dicts."""
    results = []
    for s in seeds:
        print(f"  seed={s}")
        results.append(fn(s))
    return results


def r2_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of determination (1 - SS_res/SS_tot). Returns float, not array."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred.ravel() - y_true.ravel()) ** 2)))


# ---------------------------------------------------------------------------
# Torus utilities (torus benchmark)
#
# All sampling is on the d-torus T^d = (S^1)^d, with latent state θ in [0, 2π)^d.
# OOD inputs are constructed as x = θ + 2π·k for a non-zero integer block
# k ∈ Z^d, so the wrong-DGP extrapolation problem is induced in raw coordinates,
# while the exact map φ(x) = (sin x, cos x, …) collapses OOD back onto the
# same torus as training.
# ---------------------------------------------------------------------------


def sample_torus(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly from [0, 2π)^d. Returns (n, d) array."""
    return rng.uniform(0.0, 2.0 * np.pi, size=(n, d))


def ood_torus(theta: np.ndarray, k_block: np.ndarray | int = 1) -> np.ndarray:
    """Lift a torus point θ to the period-block-k OOD coordinate x = θ + 2π·k.

    k_block = int → shift every coordinate by k periods.
    k_block = (d,) array → per-coordinate shift.
    """
    theta = np.asarray(theta)
    if np.isscalar(k_block):
        return theta + 2.0 * np.pi * float(k_block)
    return theta + 2.0 * np.pi * np.asarray(k_block, dtype=float).reshape(1, -1)


def phi_torus(x: np.ndarray) -> np.ndarray:
    """Exact periodic feature map: x ∈ R^{n×d} → (sin x_1, cos x_1, …, sin x_d, cos x_d).

    Returns (n, 2d) array. Note feature ordering is per-coordinate (sin, cos)
    pairs interleaved, matching the benchmark's φ.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    out = np.empty((n, 2 * d), dtype=x.dtype)
    out[:, 0::2] = np.sin(x)
    out[:, 1::2] = np.cos(x)
    return out


def target_g_add(theta: np.ndarray) -> np.ndarray:
    """Additive target: g_add(θ) = (1/d) Σ_j sin(θ_j)."""
    theta = np.asarray(theta)
    return np.sin(theta).mean(axis=-1)


def target_g_int(theta: np.ndarray, lam: float = 0.35) -> np.ndarray:
    """Mixed additive + multiplicative target:
        g_int(θ) = (1-λ) · g_add(θ) + λ · ∏_j sin(θ_j),   λ = 0.35 by default.
    """
    theta = np.asarray(theta)
    add  = np.sin(theta).mean(axis=-1)
    prod = np.prod(np.sin(theta), axis=-1)
    return (1.0 - lam) * add + lam * prod


def target_g_qwise(theta: np.ndarray, q: int) -> np.ndarray:
    """Normalized q-wise interaction target:
        g_{d,q}(θ) = C(d,q)^{-1/2} · Σ_{|S|=q} ∏_{j∈S} √2 sin(θ_j)

    Variance-1 under uniform θ on the torus, comparable across q.
    Iterates subsets of size q via itertools.combinations.
    """
    import itertools
    from math import comb
    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n, d = theta.shape
    if q < 1 or q > d:
        raise ValueError(f"q must be in [1, {d}], got {q}")
    sqrt2_sin = np.sqrt(2.0) * np.sin(theta)        # (n, d)
    total = np.zeros(n, dtype=theta.dtype)
    for S in itertools.combinations(range(d), q):
        total += np.prod(sqrt2_sin[:, list(S)], axis=1)
    return total / np.sqrt(comb(d, q))


def m_eff(n: int, d: int) -> float:
    """Effective coverage M_eff = N^{1/d}."""
    return float(n ** (1.0 / d))
