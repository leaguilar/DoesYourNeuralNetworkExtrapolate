"""
Exp Tor3 — Capacity at varied coverage on the periodic d-torus
================================================================

Exposes **condition (b) model-class capacity** at fixed (a)=exact φ with
**(c) coverage varied**: even when the representation is correct and per-
coordinate coverage is adequate for low-order targets, an MLP fails to
recover high-order multilinear structure as q approaches d. The failure
is rescued by extra coverage at moderate d (d=6, M=4→6) but persists at
high d (d=8, M=5).

Setup
-----
Same torus + exact-φ MLP as Tor2. Sweep the target's interaction order:
        g_{d,q}(θ) = C(d,q)^{-1/2} · Σ_{|S|=q} ∏_{j∈S} √2 sin(θ_j),  q = 1..d
which is variance-1 under uniform θ for all q.

Sweep
-----
d ∈ {4, 6, 8} × M ∈ {4, 5, 6} × q ∈ {1..d}, exact-φ only. Cells whose
N=M^d exceeds ``--n-max`` are skipped. Default N_MAX=200000 reaches
d=8/M=4 (N=65536). To reproduce the d=8/M=5/q=8 headline collapse
(N≈391k, OOD R²=0.028), pass ``--n-max 500000``.

Reproduces Figures 3, 4 and Table 2 of the torus benchmark.
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
M_LIST  = [4, 5, 6]
N_ID    = 2048
N_OOD   = 2048
N_MAX_DEFAULT = 200_000


def _train_eval_exact_phi_qwise(d: int, M: int, q: int,
                                rng: np.random.Generator, epochs: int):
    n_train = int(round(M ** d))
    theta_tr  = sample_torus(d, n_train, rng)
    y_tr      = target_g_qwise(theta_tr, q=q)
    theta_id  = sample_torus(d, N_ID, rng)
    y_id      = target_g_qwise(theta_id, q=q)
    theta_ood = sample_torus(d, N_OOD, rng)
    y_ood     = target_g_qwise(theta_ood, q=q)
    x_ood     = ood_torus(theta_ood, k_block=1)

    phi_tr  = phi_torus(theta_tr).astype(np.float32)
    phi_id  = phi_torus(theta_id).astype(np.float32)
    phi_ood = phi_torus(x_ood).astype(np.float32)

    m = MLP(in_dim=2 * d, hidden=256, layers=5)
    train_mlp(m, phi_tr, y_tr.astype(np.float32), epochs=epochs, device=DEVICE)
    yp_id  = evaluate_mlp(m, phi_id)
    yp_ood = evaluate_mlp(m, phi_ood)
    return dict(
        n_train  = n_train,
        m_eff    = m_eff(n_train, d),
        id_rmse  = rmse(yp_id,  y_id),
        ood_rmse = rmse(yp_ood, y_ood),
        id_r2    = r2_score(yp_id,  y_id),
        ood_r2   = r2_score(yp_ood, y_ood),
    )


def run(seed: int = 42, epochs: int = 3000,
        backend: str = "cpu", skip_foundation: bool = False, save: bool = False,
        d_list=None, m_list=None, q_list=None, n_max: int = N_MAX_DEFAULT):
    set_seed(seed)
    t0 = init_timing()
    rng = np.random.default_rng(seed)
    if d_list is None: d_list = D_LIST
    if m_list is None: m_list = M_LIST

    print("\n=== Exp Tor3  Capacity (q-sweep) on the d-torus ===")
    print(f"  seed={seed} epochs={epochs} N_MAX={n_max}")
    print(f"  d_list={d_list}  M_list={m_list}")

    cells = {}
    for d in d_list:
        qs = q_list if q_list is not None else list(range(1, d + 1))
        for M in m_list:
            n_target = int(round(M ** d))
            if n_target > n_max:
                print(f"  d={d:2d} M={M:2d}  skip (N={n_target} > N_max={n_max})")
                for q in qs:
                    cells[f"d={d}_M={M}_q={q}"] = dict(
                        d=d, M=M, q=q, n_train=n_target,
                        status="skipped_n_too_large",
                    )
                continue
            for q in qs:
                if q < 1 or q > d:
                    continue
                mtr = _train_eval_exact_phi_qwise(
                    d=d, M=M, q=q, rng=rng, epochs=epochs,
                )
                print(f"  d={d:2d} M={M:2d} q={q:2d}  N={mtr['n_train']:>7d}  "
                      f"M_eff={mtr['m_eff']:.3f}  "
                      f"ID R²={mtr['id_r2']:.4f}  OOD R²={mtr['ood_r2']:.4f}")
                cells[f"d={d}_M={M}_q={q}"] = dict(
                    d=d, M=M, q=q, status="ok", **mtr,
                )

    out = dict(
        experiment_id="4.3",
        name="torus_interaction_capacity",
        seed=seed,
        epochs=epochs,
        n_max=n_max,
        target=dict(name="g_qwise", normalization="C(d,q)^(-1/2) sum prod sqrt2*sin"),
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
    parser.add_argument("--q-list",          type=int, nargs="+", default=None)
    parser.add_argument("--n-max",           type=int, default=N_MAX_DEFAULT)
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, backend=args.backend,
        skip_foundation=args.skip_foundation, save=args.save,
        d_list=args.d_list, m_list=args.m_list, q_list=args.q_list,
        n_max=args.n_max)
