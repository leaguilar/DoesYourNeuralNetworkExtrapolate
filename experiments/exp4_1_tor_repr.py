"""
Exp Tor1 — Representation alignment on the periodic d-torus
=============================================================

Tests **condition (a)**: representation / feature engineering. Model class
held fixed at MLP; coverage held fixed at N ≈ 4096.

Setup
-----
Latent state θ ∈ [0, 2π)^d (the d-torus T^d = (S^1)^d). Inputs
        x = θ + 2π·k,   k ∈ Z^d
with disjoint period blocks: training uses k=0 (i.e. x = θ in [0, 2π)^d),
OOD uses k = (1, 1, …, 1) so x_ood ∈ [2π, 4π)^d. The exact map
        φ(x) = (sin x_1, cos x_1, …, sin x_d, cos x_d)
collapses OOD coordinates back onto the same torus as training, removing the
wrong-DGP extrapolation problem by construction.

Three regressors (model class held fixed):
  raw_mlp                : MLP(in_dim=d) on raw x
  exact_phi_mlp          : MLP(in_dim=2d) on φ(x)
  learned_fourier_mlp    : LearnedFourierMLP(d) — per-coordinate trainable
                           log-frequencies, init misspecified

Target
------
g_int(θ) = (1-λ) · (1/d) Σ_j sin(θ_j) + λ · ∏_j sin(θ_j),   λ = 0.35

Sweep
-----
d ∈ {1, 2, 4, 6, 8, 12}, fixed N = 4096. M_eff = 4096^(1/d) varies (rounded
in tables for readability).

Reproduces Figure 1 + Table 1 of the torus benchmark (selected rows).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    set_seed, save_result, m_eff,
    sample_torus, ood_torus, phi_torus, target_g_int,
    MLP, train_mlp, evaluate_mlp,
    rmse, r2_score, DEVICE,
    init_timing, collect_meta,
)
from models import LearnedFourierMLP


D_LIST = [1, 2, 4, 6, 8, 12]
N_TRAIN = 4096
N_ID    = 2048
N_OOD   = 2048
LAM     = 0.35


def _train_predict_mlp(model: torch.nn.Module, X_tr, y_tr, X_id, X_ood, epochs: int):
    """Train any nn.Module that maps (n, d_in)→(n, 1) on (X, y), then return
    predictions on X_id and X_ood. Reuses utils.train_mlp + evaluate_mlp.
    """
    train_mlp(model, X_tr, y_tr, epochs=epochs, device=DEVICE)
    return evaluate_mlp(model, X_id), evaluate_mlp(model, X_ood)


def _metrics(y_pred_id, y_id, y_pred_ood, y_ood):
    return dict(
        id_rmse  = rmse(y_pred_id,  y_id),
        ood_rmse = rmse(y_pred_ood, y_ood),
        id_r2    = r2_score(y_pred_id,  y_id),
        ood_r2   = r2_score(y_pred_ood, y_ood),
    )


def run(seed: int = 42, epochs: int = 3000,
        backend: str = "cpu", skip_foundation: bool = False, save: bool = False,
        d_list=None, n_train=None):
    """Tor1 sweep. backend/skip_foundation accepted for run_all.py compatibility
    but unused here (no foundation models in this experiment)."""
    set_seed(seed)
    t0 = init_timing()
    rng = np.random.default_rng(seed)
    if d_list is None:
        d_list = D_LIST
    if n_train is None:
        n_train = N_TRAIN

    print("\n=== Exp Tor1  Representation alignment on the d-torus ===")
    print(f"  seed={seed} epochs={epochs} N_train={n_train}")
    print(f"  d_list={d_list}")

    conditions = {}
    for d in d_list:
        # Training data on T^d
        theta_tr = sample_torus(d, n_train, rng)
        y_tr     = target_g_int(theta_tr, lam=LAM)
        # ID test (different sample, same distribution as training)
        theta_id = sample_torus(d, N_ID, rng)
        y_id     = target_g_int(theta_id, lam=LAM)
        # OOD: shift by k=1 in every coordinate (raw extrapolation, exact-φ trivial)
        theta_ood = sample_torus(d, N_OOD, rng)
        y_ood     = target_g_int(theta_ood, lam=LAM)
        x_ood     = ood_torus(theta_ood, k_block=1)

        # ── (a-wrong) raw MLP: input = x ──────────────────────────────────
        m_raw = MLP(in_dim=d, hidden=256, layers=5)
        yp_id_raw, yp_ood_raw = _train_predict_mlp(
            m_raw, theta_tr.astype(np.float32), y_tr.astype(np.float32),
            theta_id.astype(np.float32), x_ood.astype(np.float32),
            epochs=epochs,
        )
        cond_raw = _metrics(yp_id_raw, y_id, yp_ood_raw, y_ood)

        # ── (a-correct) exact-φ MLP: input = φ(x) ─────────────────────────
        m_exact = MLP(in_dim=2 * d, hidden=256, layers=5)
        phi_tr  = phi_torus(theta_tr).astype(np.float32)
        phi_id  = phi_torus(theta_id).astype(np.float32)
        phi_ood = phi_torus(x_ood).astype(np.float32)
        yp_id_ex, yp_ood_ex = _train_predict_mlp(
            m_exact, phi_tr, y_tr.astype(np.float32),
            phi_id, phi_ood, epochs=epochs,
        )
        cond_exact = _metrics(yp_id_ex, y_id, yp_ood_ex, y_ood)

        # ── (a-trainable) learned-Fourier MLP: input = x, ω learnable ─────
        m_lf = LearnedFourierMLP(d=d, hidden=256, layers=5,
                                 init_min=0.1, init_max=10.0,
                                 init_seed=seed)
        yp_id_lf, yp_ood_lf = _train_predict_mlp(
            m_lf, theta_tr.astype(np.float32), y_tr.astype(np.float32),
            theta_id.astype(np.float32), x_ood.astype(np.float32),
            epochs=epochs,
        )
        cond_lf = _metrics(yp_id_lf, y_id, yp_ood_lf, y_ood)
        # Also record final learned ω (one per coord) for diagnostic
        cond_lf["learned_omega"] = torch.exp(m_lf.features.log_omega).detach().cpu().numpy().tolist()

        m_val = m_eff(n_train, d)
        print(f"  d={d:2d}  M_eff≈{m_val:.2f}  "
              f"raw OOD RMSE={cond_raw['ood_rmse']:.3f}  "
              f"learned-Fourier={cond_lf['ood_rmse']:.3f}  "
              f"exact-φ={cond_exact['ood_rmse']:.3f}")

        conditions[f"d={d}"] = dict(
            d=d,
            n_train=n_train,
            m_eff=m_val,
            raw_mlp=cond_raw,
            learned_fourier_mlp=cond_lf,
            exact_phi_mlp=cond_exact,
        )

    out = dict(
        experiment_id="4.1",
        name="torus_dgp_commitment",
        seed=seed,
        epochs=epochs,
        n_train=n_train,
        target=dict(name="g_int", lam=LAM),
        ood_construction=dict(k_block=1, description="x = θ + 2π·1"),
        conditions=conditions,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--epochs",         type=int, default=3000)
    parser.add_argument("--backend",        type=str, default="cpu",
                        choices=["cpu", "gpu"])
    parser.add_argument("--skip-foundation", action="store_true")
    parser.add_argument("--save",           action="store_true")
    parser.add_argument("--d-list",         type=int, nargs="+", default=None,
                        help="Override the d sweep grid (smoke test)")
    parser.add_argument("--n-train",        type=int, default=None,
                        help="Override N_train (smoke test)")
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, backend=args.backend,
        skip_foundation=args.skip_foundation, save=args.save,
        d_list=args.d_list, n_train=args.n_train)
