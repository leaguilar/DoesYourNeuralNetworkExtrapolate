"""
Exp 4.4 — Tor-Noise: noise robustness of the DGP commitment on the d-torus
==========================================================================

Tests **Prop:erm (ERM blindness) and Lem:canon (constructive existence)** on the
torus benchmark. Coverage and capacity are non-binding (M_eff ≫ M_crit ≈ 4 at
both d=2 and d=4); the only thing varying is the **commitment** (raw vs.
learned-Fourier vs. exact-φ) and the **noise level σ** on training y.

Setup
-----
Same DGP as Exp 4.1 (Tor-Repr): θ ∈ [0, 2π)^d uniform, target
    g_int(θ) = (1-λ) · (1/d) Σ_j sin(θ_j) + λ · ∏_j sin(θ_j),  λ = 0.35,
training inputs x = θ on T^d (k=0), OOD inputs x = θ + 2π·1 (k=1).
Training y has additive Gaussian noise, ID/OOD targets remain clean (oracle):

    y_train = g_int(θ_tr) + N(0, σ²)
    y_id    = g_int(θ_id)             # clean (oracle reference)
    y_ood   = g_int(θ_ood)            # clean (oracle reference)

Three regressors (model class held fixed):
  raw_mlp                : MLP(in_dim=d) on raw x
  exact_phi_mlp          : MLP(in_dim=2d) on φ(x)
  learned_fourier_mlp    : LearnedFourierMLP(d) — per-coordinate trainable ω,
                           init misspecified

Sweep
-----
σ ∈ {0, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0}   (8 points spanning 4 OOMs)
d ∈ {4, 6, 8}                                       (matches Tor-Cov / Tor-Cap)
N_train = 4096                                      (matches Exp 4.1)

The σ grid spans both regimes of Prop:erm:
  - σ ≪ ε√n/2 (≈ 1.6 here for ε ≈ 0.05, n = 4096): bound is vacuous (signal-dominated)
    {σ = 0.01, 0.1, 0.3, 1.0}
  - σ ≥ ε√n/2: bound is informative (noise-dominated, ERM blindness provable)
    {σ = 3.0, 10.0, 30.0}

Predicted (Prop:erm + Lem:canon)
--------------------------------
- raw_mlp:        OOD ≈ const in σ (commitment-bias dominated, σ-independent)
- exact_phi_mlp:  OOD scales ≈ σ   (Bayes risk only)
- learned_fourier_mlp: depends on whether optimiser recovers ω
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    set_seed, save_result,
    sample_torus, ood_torus, phi_torus, target_g_int,
    MLP, train_mlp, evaluate_mlp,
    rmse, r2_score, DEVICE,
    init_timing, collect_meta,
)
from models import LearnedFourierMLP


SIGMA_LIST_DEFAULT = (0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
D_LIST_DEFAULT     = (4, 6, 8)
N_TRAIN  = 4096
N_ID     = 2048
N_OOD    = 2048
LAM      = 0.35


def _train_predict_mlp(model: torch.nn.Module, X_tr, y_tr, X_id, X_ood, epochs: int):
    train_mlp(model, X_tr, y_tr, epochs=epochs, device=DEVICE)
    return evaluate_mlp(model, X_id), evaluate_mlp(model, X_ood)


def _metrics(y_pred_id, y_id, y_pred_ood, y_ood):
    return dict(
        id_rmse  = rmse(y_pred_id,  y_id),
        ood_rmse = rmse(y_pred_ood, y_ood),
        id_r2    = r2_score(y_pred_id,  y_id),
        ood_r2   = r2_score(y_pred_ood, y_ood),
    )


def _bayes_slope(sigmas, ood_rmse_per_sigma) -> float:
    """OLS slope of log(OOD RMSE) vs log(σ) over σ > 0 cells.

    Theory predicts ≈ 1 for the canonical (exact-φ) commitment: OOD error is
    Bayes-only and scales linearly with σ. Returns nan if fewer than 2 σ>0
    cells are available."""
    s = np.asarray(sigmas, dtype=float)
    e = np.asarray(ood_rmse_per_sigma, dtype=float)
    mask = (s > 0) & np.isfinite(e) & (e > 0)
    if mask.sum() < 2:
        return float("nan")
    log_s = np.log(s[mask])
    log_e = np.log(e[mask])
    slope, _ = np.polyfit(log_s, log_e, 1)
    return float(slope)


def run(seed: int = 42, epochs: int = 3000,
        backend: str = "cpu", skip_foundation: bool = False, save: bool = False,
        d_list=None, sigma_list=None, n_train=None):
    """Tor-Noise σ × d sweep. backend/skip_foundation accepted for run_all.py
    compatibility but unused."""
    set_seed(seed)
    t0 = init_timing()
    rng = np.random.default_rng(seed)
    if d_list is None:
        d_list = D_LIST_DEFAULT
    if sigma_list is None:
        sigma_list = SIGMA_LIST_DEFAULT
    if n_train is None:
        n_train = N_TRAIN

    print("\n=== Exp 4.4 Tor-Noise ===")
    print(f"  seed={seed} epochs={epochs} N_train={n_train}")
    print(f"  d_list={list(d_list)}  sigma_list={list(sigma_list)}")

    conditions = {}
    bayes_slopes = {}

    for d in d_list:
        # Reuse the same θ samples across σ to isolate the noise effect.
        theta_tr  = sample_torus(d, n_train, rng)
        theta_id  = sample_torus(d, N_ID, rng)
        theta_ood = sample_torus(d, N_OOD, rng)
        y_tr_clean = target_g_int(theta_tr,  lam=LAM)
        y_id       = target_g_int(theta_id,  lam=LAM)
        y_ood      = target_g_int(theta_ood, lam=LAM)
        x_ood      = ood_torus(theta_ood, k_block=1)
        phi_tr_arr  = phi_torus(theta_tr).astype(np.float32)
        phi_id_arr  = phi_torus(theta_id).astype(np.float32)
        phi_ood_arr = phi_torus(x_ood).astype(np.float32)

        ood_raw_per_sigma  = []
        ood_lf_per_sigma   = []
        ood_ex_per_sigma   = []

        for sigma in sigma_list:
            # Resample noise per σ but keep the seed deterministic via rng state.
            noise = rng.normal(0.0, float(sigma), size=y_tr_clean.shape)
            y_tr  = (y_tr_clean + noise).astype(np.float32)

            # ── (a-wrong) raw MLP ─────────────────────────────────────────
            m_raw = MLP(in_dim=d, hidden=256, layers=5)
            yp_id_raw, yp_ood_raw = _train_predict_mlp(
                m_raw, theta_tr.astype(np.float32), y_tr,
                theta_id.astype(np.float32), x_ood.astype(np.float32),
                epochs=epochs,
            )
            cond_raw = _metrics(yp_id_raw, y_id, yp_ood_raw, y_ood)

            # ── (a-correct) exact-φ MLP ───────────────────────────────────
            m_exact = MLP(in_dim=2 * d, hidden=256, layers=5)
            yp_id_ex, yp_ood_ex = _train_predict_mlp(
                m_exact, phi_tr_arr, y_tr, phi_id_arr, phi_ood_arr,
                epochs=epochs,
            )
            cond_exact = _metrics(yp_id_ex, y_id, yp_ood_ex, y_ood)

            # ── (a-trainable) learned-Fourier MLP ─────────────────────────
            m_lf = LearnedFourierMLP(d=d, hidden=256, layers=5,
                                     init_min=0.1, init_max=10.0,
                                     init_seed=seed)
            yp_id_lf, yp_ood_lf = _train_predict_mlp(
                m_lf, theta_tr.astype(np.float32), y_tr,
                theta_id.astype(np.float32), x_ood.astype(np.float32),
                epochs=epochs,
            )
            cond_lf = _metrics(yp_id_lf, y_id, yp_ood_lf, y_ood)
            cond_lf["learned_omega"] = torch.exp(
                m_lf.features.log_omega).detach().cpu().numpy().tolist()

            # ── in-window systematic gap between raw and exact-φ predictions
            # on a clean grid: the empirical stand-in for the ε_W of Prop:erm.
            # `eps_w_observed` is the L²(D_id) norm that Def:obseq and Prop:erm
            # are stated in; `eps_observed` is the legacy mean-absolute (L¹)
            # value, kept only so older result files stay comparable.
            gap_id       = yp_id_raw - yp_id_ex
            eps_w_observed = float(np.sqrt(np.mean(gap_id ** 2)))
            eps_observed   = float(np.mean(np.abs(gap_id)))

            print(f"  d={d:2d} σ={sigma:>6.3f}  "
                  f"raw OOD={cond_raw['ood_rmse']:.3f}  "
                  f"learned-Fourier={cond_lf['ood_rmse']:.3f}  "
                  f"exact-φ={cond_exact['ood_rmse']:.3f}  "
                  f"ε_W={eps_w_observed:.3f}")

            conditions[f"d={d}_sigma={sigma}"] = dict(
                d=d,
                sigma=float(sigma),
                noise_std=float(sigma),
                n_train=n_train,
                raw_mlp=cond_raw,
                learned_fourier_mlp=cond_lf,
                exact_phi_mlp=cond_exact,
                eps_w_observed=eps_w_observed,
                eps_observed=eps_observed,
            )

            ood_raw_per_sigma.append(cond_raw["ood_rmse"])
            ood_lf_per_sigma.append(cond_lf["ood_rmse"])
            ood_ex_per_sigma.append(cond_exact["ood_rmse"])

        # Per-d Bayes-slope diagnostic across σ>0.
        bayes_slopes[f"d={d}"] = dict(
            raw_mlp             = _bayes_slope(sigma_list, ood_raw_per_sigma),
            learned_fourier_mlp = _bayes_slope(sigma_list, ood_lf_per_sigma),
            exact_phi_mlp       = _bayes_slope(sigma_list, ood_ex_per_sigma),
        )
        bs = bayes_slopes[f"d={d}"]
        print(f"  d={d:2d}  log-log OOD slope vs σ:  "
              f"raw={bs['raw_mlp']:.2f}  "
              f"learned-Fourier={bs['learned_fourier_mlp']:.2f}  "
              f"exact-φ={bs['exact_phi_mlp']:.2f}  (theory: ≈0, ≈?, ≈1)")

    out = dict(
        experiment_id="4.4",
        name="torus_noise_robustness",
        seed=seed,
        epochs=epochs,
        n_train=n_train,
        target=dict(name="g_int", lam=LAM),
        ood_construction=dict(k_block=1, description="x = θ + 2π·1"),
        sigma_list=[float(s) for s in sigma_list],
        d_list=[int(d) for d in d_list],
        conditions=conditions,
        bayes_slopes=bayes_slopes,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--epochs",          type=int, default=3000)
    parser.add_argument("--backend",         type=str, default="cpu",
                        choices=["cpu", "gpu"])
    parser.add_argument("--skip-foundation", action="store_true")
    parser.add_argument("--save",            action="store_true")
    parser.add_argument("--d-list",          type=int,   nargs="+", default=None,
                        help="Override the d sweep grid (smoke test).")
    parser.add_argument("--sigma-list",      type=float, nargs="+", default=None,
                        help="Override the σ sweep grid (smoke test).")
    parser.add_argument("--n-train",         type=int, default=None,
                        help="Override N_train (smoke test).")
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, backend=args.backend,
        skip_foundation=args.skip_foundation, save=args.save,
        d_list=args.d_list, sigma_list=args.sigma_list, n_train=args.n_train)
