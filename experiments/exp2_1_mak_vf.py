#!/usr/bin/env python3
"""
Exp MAK-VF — Mass-Action Kinetics vector field
==============================================

CANONICAL MASS-ACTION-KINETICS SCENARIO ON A SYNTHETIC 2-NODE GRAPH.

The dynamics, parameter values, and 2-node-graph structure are taken
directly from the canonical formulation of MAK on networks introduced by
Vasiliauskaite & Antulov-Fantulin (2024), Commun. Phys. 7:348
(DOI: 10.1038/s42005-024-01837-w), Eq. 10:

    ẋ_i = F - B·x_i - R · sum_j A_ij · x_i · x_j

For a fully-connected 2-node graph (A_12 = A_21 = 1, A_11 = A_22 = 0):

    ẋ_1 = F - B·x_1 - R·x_1·x_2      (= 0.5 - 0.1·x_1 - x_1·x_2)
    ẋ_2 = F - B·x_2 - R·x_2·x_1

Parameters: B=0.1, R=1, F=0.5 — textbook values for the canonical MAK
demonstration in Vasiliauskaite & Antulov-Fantulin (2024). They are
representative of the dimensionless CSTR mass-action regime: F is the
constant inflow (k_0 c_in), B is the first-order outflow (k_0), R is the
bimolecular rate constant. The dimensionless values map to typical orders
of magnitude in real reactors (k_0 ~ 1e-2 s^-1, k_bimol ~ 1e-2 M^-1 s^-1
as in the chlorite-sulfite mechanism of Frerichs et al. 2001) when rescaled
to unit time and unit concentration. We do NOT claim these values are
fitted to a specific real reactor; the experiment is a controlled
demonstration of the framework's claim using a published canonical MAK
formulation.

Real-world precedents for MAK as a DGP class:
    Vasiliauskaite & Antulov-Fantulin (2024) — MAK on real network dynamics
        (the base paper this experiment instantiates).
    Frerichs et al. (2001), DOI 10.1021/jp003363u — chlorite-sulfite CSTR
        oscillator, real measured (pH, Pt) dynamics with MAK rate constants
        in Tables 1-4.
    Field & Noyes (1974), DOI 10.1063/1.1681288 — Oregonator, the
        canonical bimolecular MAK model of the Belousov-Zhabotinsky reaction.

Key claim (feature engineering as DGP commitment):
    Features [1, x_i, x_i·x_j] make ẋ_i GLOBALLY LINEAR in feature space
    with fixed coefficients w = [F, -B, -R].  OLS recovers these exactly
    from any training region → SC_gap = 0 → perfect OOD generalization.

    Features [x_1, x_2] cannot represent the bilinear interaction → any
    model learns a local approximation valid only near the training data
    → SC_gap > 0 outside the training region → OOD failure.

    Specifying the MAK ODE is equivalent to choosing x_i·x_j as a feature.
    This mirrors Figure 1 of Vasiliauskaite & Antulov-Fantulin (2024).

Conditions:
    raw_mlp   — MLP(x_1, x_2) → (ẋ_1, ẋ_2):        no DGP knowledge
    mak_ols   — OLS([1, x_i, x_i·x_j]) → ẋ_i:        exact DGP feature
    mak_mlp   — MLP([x_i, x_i·x_j]) → ẋ_i:           DGP feature + neural net

Usage:
    conda run -n data_analysis python src/experiments/exp_mak_vf.py --save --figure
    conda run -n data_analysis python src/experiments/exp_mak_vf.py --seed 1 --save
"""

import argparse
import os
import sys

import numpy as np
import pysindy as ps

sys.path.insert(0, os.path.dirname(__file__))
from utils import (MLP, evaluate_mlp, ood_relative_error, save_result,
                   set_seed, train_mlp, init_timing, collect_meta)


# ---------------------------------------------------------------------------
# SINDy δ_ood diagnostic helper (matches exp2_2_kepler:_sindy_static).
# ---------------------------------------------------------------------------

def _sindy_delta(Z_tr, y_tr, Z_id, Z_ood, y_id, y_ood,
                 degree=2, threshold=0.05):
    """Sparse polynomial regression y = Theta(Z) @ xi via pysindy.

    Returns (id_error, ood_error, xi, n_terms, delta_train, delta_ood):
      - id/ood_error: relative error in raw target space.
      - delta_*: squared-NMSE residual in the SINDy fit space, matching
        the §3.3 toy-battery formula (exp1_4_sindy_coord.py:127-130).
    """
    lib = ps.PolynomialLibrary(degree=degree, include_bias=True)
    lib.fit(Z_tr)
    Th_tr = np.asarray(lib.transform(Z_tr))
    Th_id = np.asarray(lib.transform(Z_id))
    Th_oo = np.asarray(lib.transform(Z_ood))

    opt = ps.STLSQ(threshold=threshold, alpha=0.0, max_iter=50)
    opt.fit(Th_tr, y_tr.reshape(-1, 1))
    xi = np.asarray(opt.coef_[0])

    pred_tr  = Th_tr @ xi
    pred_id  = Th_id @ xi
    pred_ood = Th_oo @ xi

    delta_train = float(np.sum((y_tr  - pred_tr ) ** 2)
                        / (np.sum(y_tr  ** 2) + 1e-12))
    delta_ood   = float(np.sum((y_ood - pred_ood) ** 2)
                        / (np.sum(y_ood ** 2) + 1e-12))

    ie = ood_relative_error(pred_id,  y_id)
    oe = ood_relative_error(pred_ood, y_ood)
    n_terms = int((np.abs(xi) > 1e-10).sum())
    return ie, oe, xi.tolist(), n_terms, delta_train, delta_ood

# ---------------------------------------------------------------------------
# MAK dynamics  (Vasiliauskaite & Antulov-Fantulin 2024, Eq. 10)
# 2-node undirected graph: A_12 = A_21 = 1, A_11 = A_22 = 0
# ---------------------------------------------------------------------------
_B, _R, _F = 0.1, 1.0, 0.5   # degradation, interaction rate, production rate


def mak_deriv(x1, x2):
    """True MAK vector field for the 2-node graph."""
    dx1 = _F - _B * x1 - _R * x1 * x2
    dx2 = _F - _B * x2 - _R * x2 * x1   # symmetric (x1·x2 = x2·x1)
    return dx1.astype(np.float32), dx2.astype(np.float32)


def feat_mak(xi, xj):
    """MAK feature vector for node i: [1, x_i, x_i·x_j]."""
    return np.stack([np.ones_like(xi), xi, xi * xj], axis=1)


def ols_pred(X_tr, y_tr, X_te):
    w, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    return X_te @ w


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(seed=42, n_train=300, train_center=0.8, train_half=0.4,
        state_max=2.0, epochs=3000, save=False, figure=False):
    set_seed(seed)
    t0 = init_timing()
    rng = np.random.default_rng(seed)

    lo, hi = train_center - train_half, train_center + train_half

    # Training data: i.i.d. uniform in [lo, hi]² (centred square)
    x1_tr = rng.uniform(lo, hi, n_train).astype(np.float32)
    x2_tr = rng.uniform(lo, hi, n_train).astype(np.float32)
    dx1_tr, dx2_tr = mak_deriv(x1_tr, x2_tr)

    # ID evaluation: fresh samples from the same training region
    N_EVAL = 500
    rng_eval = np.random.default_rng(seed + 1000)
    x1_id = rng_eval.uniform(lo, hi, N_EVAL).astype(np.float32)
    x2_id = rng_eval.uniform(lo, hi, N_EVAL).astype(np.float32)
    dx1_id, dx2_id = mak_deriv(x1_id, x2_id)

    # OOD evaluation: rejection-sample from [0, state_max]² outside training square
    _x1 = rng_eval.uniform(0.0, state_max, N_EVAL * 20).astype(np.float32)
    _x2 = rng_eval.uniform(0.0, state_max, N_EVAL * 20).astype(np.float32)
    ood_mask = ~((_x1 >= lo) & (_x1 <= hi) & (_x2 >= lo) & (_x2 <= hi))
    x1_ood = _x1[ood_mask][:N_EVAL]
    x2_ood = _x2[ood_mask][:N_EVAL]
    dx1_ood, dx2_ood = mak_deriv(x1_ood, x2_ood)

    true_id  = np.stack([dx1_id,  dx2_id],  axis=1)
    true_ood = np.stack([dx1_ood, dx2_ood], axis=1)

    results = {}

    # ── raw_mlp: MLP(x_1, x_2) → (ẋ_1, ẋ_2) ─────────────────────────────
    X_raw_tr = np.stack([x1_tr, x2_tr], axis=1)
    y_raw_tr = np.stack([dx1_tr, dx2_tr], axis=1)
    mlp_raw  = MLP(in_dim=2, hidden=128, layers=4, out_dim=2)
    train_mlp(mlp_raw, X_raw_tr, y_raw_tr, epochs=epochs, lr=1e-3)

    p_id_raw  = evaluate_mlp(mlp_raw, np.stack([x1_id,  x2_id],  axis=1))
    p_ood_raw = evaluate_mlp(mlp_raw, np.stack([x1_ood, x2_ood], axis=1))
    if p_id_raw.ndim == 1:
        p_id_raw  = p_id_raw.reshape(-1, 2)
        p_ood_raw = p_ood_raw.reshape(-1, 2)

    results["raw_mlp"] = dict(
        id_error  = ood_relative_error(p_id_raw.flatten(),  true_id.flatten()),
        ood_error = ood_relative_error(p_ood_raw.flatten(), true_ood.flatten()),
    )

    # ── mak_ols: OLS([1, x_i, x_i·x_j]) → ẋ_i ───────────────────────────
    F1_tr = feat_mak(x1_tr, x2_tr)   # node-1 design matrix
    F2_tr = feat_mak(x2_tr, x1_tr)   # node-2 design matrix (symmetric)

    p_id_ols  = np.stack([ols_pred(F1_tr, dx1_tr, feat_mak(x1_id,  x2_id)),
                           ols_pred(F2_tr, dx2_tr, feat_mak(x2_id,  x1_id))], axis=1)
    p_ood_ols = np.stack([ols_pred(F1_tr, dx1_tr, feat_mak(x1_ood, x2_ood)),
                           ols_pred(F2_tr, dx2_tr, feat_mak(x2_ood, x1_ood))], axis=1)

    results["mak_ols"] = dict(
        id_error  = ood_relative_error(p_id_ols.flatten(),  true_id.flatten()),
        ood_error = ood_relative_error(p_ood_ols.flatten(), true_ood.flatten()),
    )

    # ── mak_mlp: MLP([x_i, x_i·x_j]) → ẋ_i ──────────────────────────────
    xi_all  = np.concatenate([x1_tr,  x2_tr]).astype(np.float32)
    xj_all  = np.concatenate([x2_tr,  x1_tr]).astype(np.float32)
    dxi_all = np.concatenate([dx1_tr, dx2_tr]).astype(np.float32)
    X_mak_tr = np.stack([xi_all, xi_all * xj_all], axis=1)

    mlp_mak = MLP(in_dim=2, hidden=128, layers=4, out_dim=1)
    train_mlp(mlp_mak, X_mak_tr, dxi_all, epochs=epochs, lr=1e-3)

    def _pred_mak_mlp(xi, xj):
        X = np.stack([xi, xi * xj], axis=1).astype(np.float32)
        return evaluate_mlp(mlp_mak, X)

    p_id_mml  = np.stack([_pred_mak_mlp(x1_id,  x2_id),
                           _pred_mak_mlp(x2_id,  x1_id)], axis=1)
    p_ood_mml = np.stack([_pred_mak_mlp(x1_ood, x2_ood),
                           _pred_mak_mlp(x2_ood, x1_ood)], axis=1)

    results["mak_mlp"] = dict(
        id_error  = ood_relative_error(p_id_mml.flatten(),  true_id.flatten()),
        ood_error = ood_relative_error(p_ood_mml.flatten(), true_ood.flatten()),
    )

    # ── SINDy δ_ood diagnostic conditions ─────────────────────────────────
    # Training/eval rows: stack both nodes (each node fits the same dynamics
    # by symmetry of the 2-node graph). Each row is (x_i, x_j) → ẋ_i.
    Z_raw_tr  = np.column_stack([xi_all, xj_all])
    Z_raw_id  = np.column_stack([np.concatenate([x1_id,  x2_id]),
                                  np.concatenate([x2_id,  x1_id])])
    Z_raw_ood = np.column_stack([np.concatenate([x1_ood, x2_ood]),
                                  np.concatenate([x2_ood, x1_ood])])
    y_id_all  = np.concatenate([dx1_id,  dx2_id])
    y_ood_all = np.concatenate([dx1_ood, dx2_ood])

    # Canonical: raw (x_i, x_j) with degree-2 polynomial library — the library
    # contains x_i·x_j so the bilinear MAK dynamics are recovered exactly.
    ie, oe, xi_, k, dtr, dood = _sindy_delta(
        Z_raw_tr, dxi_all, Z_raw_id, Z_raw_ood, y_id_all, y_ood_all,
        degree=2, threshold=0.05)
    results["raw_xixj_sindy"] = dict(
        id_error=ie, ood_error=oe, xi=xi_, n_terms=k,
        delta_train=dtr, delta_ood=dood,
    )

    # Wrong basis: tanh(x_i, x_j) with degree-2 polynomial library — atanh is
    # non-polynomial, so the bilinear x_i·x_j cannot be expressed in tanh-space.
    Z_tanh_tr  = np.tanh(Z_raw_tr)
    Z_tanh_id  = np.tanh(Z_raw_id)
    Z_tanh_ood = np.tanh(Z_raw_ood)
    ie, oe, xi_, k, dtr, dood = _sindy_delta(
        Z_tanh_tr, dxi_all, Z_tanh_id, Z_tanh_ood, y_id_all, y_ood_all,
        degree=2, threshold=0.05)
    results["tanh_xixj_sindy"] = dict(
        id_error=ie, ood_error=oe, xi=xi_, n_terms=k,
        delta_train=dtr, delta_ood=dood,
    )

    # Missing variable: x_i alone with degree-2 polynomial library — cannot
    # express the x_i·x_j interaction; best fit absorbs E[x_j] into the slope.
    ie, oe, xi_, k, dtr, dood = _sindy_delta(
        Z_raw_tr[:, :1], dxi_all,
        Z_raw_id[:, :1], Z_raw_ood[:, :1],
        y_id_all, y_ood_all,
        degree=2, threshold=0.05)
    results["univariate_xi_sindy"] = dict(
        id_error=ie, ood_error=oe, xi=xi_, n_terms=k,
        delta_train=dtr, delta_ood=dood,
    )

    # ── Print ──────────────────────────────────────────────────────────────
    w1, _, _, _ = np.linalg.lstsq(F1_tr, dx1_tr, rcond=None)
    print(f"\n=== Exp MAK-VF  (seed={seed}) ===")
    print(f"  DGP: ẋ_i = F - B·x_i - R·x_i·x_j   "
          f"(F={_F}, B={_B}, R={_R})")
    print(f"  mak_ols recovered: "
          f"w_0={w1[0]:.4f} (true F={_F}),  "
          f"w_1={w1[1]:.4f} (true -B={-_B}),  "
          f"w_2={w1[2]:.4f} (true -R={-_R})")
    print(f"  Training region: [{lo:.2f}, {hi:.2f}]²   "
          f"OOD: [0, {state_max}]² \\ [{lo:.2f}, {hi:.2f}]²")
    print(f"\n  {'condition':<12}  {'ID err%':>8}  {'OOD err%':>9}")
    for cond, res in results.items():
        print(f"  {cond:<12}  {res['id_error']:>8.2f}  {res['ood_error']:>9.2f}")

    out = dict(
        experiment_id="2.1",
        name="mak_vector_field",
        seed=seed,
        config=dict(B=_B, R=_R, F=_F,
                    train_center=train_center, train_half=train_half,
                    state_max=state_max, n_train=n_train),
        conditions=results,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)

    if figure:
        artefacts = _save_figure_artefacts(
            x1_tr, x2_tr, mlp_raw, F1_tr, dx1_tr, F2_tr, dx2_tr,
            train_center=train_center, train_half=train_half,
            state_max=state_max)
        # Build the figure now so the experiment's --figure flag still produces
        # the PDF in one shot. The plot module is reusable on its own via
        # `python scripts/main_plot_concept.py`.
        _scripts_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from figures.main.mak_vf import build as build_mak_vf  # noqa: E402
        build_mak_vf(data=artefacts)
    return out


# ---------------------------------------------------------------------------
# Figure artefacts  (consumed by scripts/figures/main/mak_vf.py, which
# mirrors Fig 1 of Vasiliauskaite & Antulov-Fantulin 2024)
# ---------------------------------------------------------------------------

def _save_figure_artefacts(x1_tr, x2_tr, mlp_raw,
                           F1_tr, dx1_tr, F2_tr, dx2_tr,
                           train_center=0.8, train_half=0.4, state_max=2.0):
    """Compute grid-evaluated MAK vector-field predictions and save to npz.

    Path: results/expmak_vf/figure_artefacts.npz.
    Returns the artefact dict so the caller can pass it to the plot module
    directly without a round-trip through disk.

    streamplot requires a 1-D coordinate array and 2-D u/v arrays where
    u[row, col] is the x-component at (x[col], y[row]).
    """
    N_GRID = 100
    g = np.linspace(0.0, state_max, N_GRID)
    G1, G2 = np.meshgrid(g, g)
    g1 = G1.ravel().astype(np.float32)
    g2 = G2.ravel().astype(np.float32)

    u_t, v_t = mak_deriv(g1, g2)

    pred_raw = evaluate_mlp(mlp_raw, np.stack([g1, g2], axis=1))
    if pred_raw.ndim == 1:
        pred_raw = pred_raw.reshape(-1, 2)
    u_r = pred_raw[:, 0].astype(np.float32)
    v_r = pred_raw[:, 1].astype(np.float32)

    u_o = ols_pred(F1_tr, dx1_tr, feat_mak(g1, g2)).astype(np.float32)
    v_o = ols_pred(F2_tr, dx2_tr, feat_mak(g2, g1)).astype(np.float32)

    def r2d(a):
        return a.reshape(N_GRID, N_GRID)

    artefacts = dict(
        x1_tr=np.asarray(x1_tr, dtype=np.float32),
        x2_tr=np.asarray(x2_tr, dtype=np.float32),
        g=g.astype(np.float32),
        U_t=r2d(u_t),  V_t=r2d(v_t),
        U_r=r2d(u_r),  V_r=r2d(v_r),
        U_o=r2d(u_o),  V_o=r2d(v_o),
        state_max=np.float32(state_max),
        train_center=np.float32(train_center),
        train_half=np.float32(train_half),
    )

    out_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "expmak_vf"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figure_artefacts.npz")
    np.savez(out_path, **artefacts)
    print(f"  Figure artefacts → {out_path}")
    return artefacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n-train",      type=int,   default=300)
    parser.add_argument("--train-center", type=float, default=0.8)
    parser.add_argument("--train-half",   type=float, default=0.4)
    parser.add_argument("--state-max",    type=float, default=2.0)
    parser.add_argument("--epochs",       type=int,   default=3000)
    parser.add_argument("--save",         action="store_true")
    parser.add_argument("--figure",       action="store_true")
    args = parser.parse_args()

    run(seed=args.seed,
        n_train=args.n_train,
        train_center=args.train_center,
        train_half=args.train_half,
        state_max=args.state_max,
        epochs=args.epochs,
        save=args.save,
        figure=args.figure)
