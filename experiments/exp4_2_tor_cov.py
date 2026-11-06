"""
Exp Tor2 — Critical coverage on the periodic d-torus
======================================================

Tests **condition (c)**: data coverage of the representation space. (a) and
(b) held fixed (exact φ + MLP); only N (the coverage budget) is varied.

Setup
-----
Same torus DGP as Tor1, target g_qwise(θ, q=d) (the full-order normalized
interaction target — variance 1 under uniform θ for all d, no additive
shortcut), exact feature map
        φ(x) = (sin x_1, cos x_1, …, sin x_d, cos x_d).
For each (d, M) cell, draw N = round(M^d) iid training samples on T^d
(directly varying coverage M = N^{1/d}). Train an MLP on φ(x_train) and
evaluate ID/OOD as in Tor1.

Note: We use the full-order interaction g_qwise(q=d) rather than the mixed
g_int because g_int is dominated by its additive component at high d (the
multiplicative term carries weight $(0.5)^d$ of total variance). The
full-order target keeps the coverage bottleneck visible across d.

Sweep
-----
d ∈ {4, 6, 8} × M ∈ {2, 3, 4, 5, 6}, exact-φ only. Cells whose N exceeds
``--n-max`` are skipped. Default N_MAX=200000 keeps d ∈ {4, 6} fully
covered and d=8 covered up to M=4. To reproduce the d=8/M=5 cell
(N≈391k), pass ``--n-max 500000``.

Reproduces Figure 2 of the torus benchmark.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    set_seed, save_result, m_eff,
    sample_torus, ood_torus, phi_torus, target_g_qwise,
    MLP, train_mlp, evaluate_mlp,
    rmse, r2_score, DEVICE,
    init_timing, collect_meta,
)


D_LIST  = [4, 6, 8]
M_LIST  = [2, 3, 4, 5, 6]
N_ID    = 2048
N_OOD   = 2048
N_MAX_DEFAULT = 200_000


def _train_eval_exact_phi(theta_tr, y_tr, theta_id, y_id, x_ood, y_ood,
                          d: int, epochs: int):
    phi_tr  = phi_torus(theta_tr).astype(np.float32)
    phi_id  = phi_torus(theta_id).astype(np.float32)
    phi_ood = phi_torus(x_ood).astype(np.float32)
    m = MLP(in_dim=2 * d, hidden=256, layers=5)
    train_mlp(m, phi_tr, y_tr.astype(np.float32), epochs=epochs, device=DEVICE)
    yp_id  = evaluate_mlp(m, phi_id)
    yp_ood = evaluate_mlp(m, phi_ood)
    return dict(
        id_rmse  = rmse(yp_id,  y_id),
        ood_rmse = rmse(yp_ood, y_ood),
        id_r2    = r2_score(yp_id,  y_id),
        ood_r2   = r2_score(yp_ood, y_ood),
    )


def run(seed: int = 42, epochs: int = 3000,
        backend: str = "cpu", skip_foundation: bool = False, save: bool = False,
        d_list=None, m_list=None, n_max: int = N_MAX_DEFAULT):
    set_seed(seed)
    t0 = init_timing()
    rng = np.random.default_rng(seed)
    if d_list is None: d_list = D_LIST
    if m_list is None: m_list = M_LIST

    print("\n=== Exp Tor2  Critical coverage on the d-torus ===")
    print(f"  seed={seed} epochs={epochs} N_MAX={n_max}")
    print(f"  d_list={d_list}  M_list={m_list}")

    cells = {}
    for d in d_list:
        for M in m_list:
            n_train = int(round(M ** d))
            if n_train > n_max:
                print(f"  d={d:2d} M={M:2d}  skip (N={n_train} > N_max={n_max})")
                cells[f"d={d}_M={M}"] = dict(
                    d=d, M=M, n_train=n_train, status="skipped_n_too_large",
                )
                continue

            theta_tr  = sample_torus(d, n_train, rng)
            y_tr      = target_g_qwise(theta_tr, q=d)
            theta_id  = sample_torus(d, N_ID, rng)
            y_id      = target_g_qwise(theta_id, q=d)
            theta_ood = sample_torus(d, N_OOD, rng)
            y_ood     = target_g_qwise(theta_ood, q=d)
            x_ood     = ood_torus(theta_ood, k_block=1)

            mtr = _train_eval_exact_phi(
                theta_tr, y_tr, theta_id, y_id, x_ood, y_ood,
                d=d, epochs=epochs,
            )
            m_actual = m_eff(n_train, d)
            print(f"  d={d:2d} M={M:2d}  N={n_train:>7d}  M_eff={m_actual:.3f}  "
                  f"OOD RMSE={mtr['ood_rmse']:.4f}  OOD R²={mtr['ood_r2']:.4f}")
            cells[f"d={d}_M={M}"] = dict(
                d=d, M=M, n_train=n_train, m_eff_actual=m_actual,
                status="ok", **mtr,
            )

    out = dict(
        experiment_id="4.2",
        name="torus_critical_coverage",
        seed=seed,
        epochs=epochs,
        n_max=n_max,
        target=dict(name="g_qwise", q="d_full_order"),
        cells=cells,
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
    parser.add_argument("--d-list",          type=int, nargs="+", default=None)
    parser.add_argument("--m-list",          type=int, nargs="+", default=None)
    parser.add_argument("--n-max",           type=int, default=N_MAX_DEFAULT)
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, backend=args.backend,
        skip_foundation=args.skip_foundation, save=args.save,
        d_list=args.d_list, m_list=args.m_list, n_max=args.n_max)
