"""
Rebuttal E3 — misspecification / partial-correctness robustness of the phi commitment
======================================================================================
Answers reviewer b1cC Q3 ("how robust is the approach to a partially-correct /
misspecified feature map?") on the paper's torus benchmark infrastructure
(exp4_*: latent theta on T^d, train k=0 block, OOD x = theta + 2pi·k=1 block;
model class fixed at the repo-standard tanh MLP 256x5, epochs 3000, N=4096).

Three misspecification axes (OOD error as a CURVE vs degree — graceful vs cliff):

  A. wrong Fourier frequency   phi_omega(x) = (sin omega·x_j, cos omega·x_j),
                               omega/omega_true in {0.90 … 1.50}; omega_true = 1.
                               raw-MLP (no phi) included as the "fully wrong" anchor.
  B. partial phi               phi applied to only the first m of d=4 coordinates
                               (raw pass-through for the rest), m = 0 … 4.
                               m=4 exact, m=0 raw — partial correctness axis.
  C. noisy phi                 exact frequencies but the map is computed with
                               additive feature noise N(0, s^2), fresh at train
                               and eval (a corrupted sensor / imperfect map),
                               s in {0 … 0.4}.

Target: g_int(theta) = 0.65·mean_j sin(theta_j) + 0.35·prod_j sin(theta_j)
(std ≈ 0.47 for d=2, ≈ 0.34 for d=4) — same as exp4_1. Metrics: ID/OOD RMSE + R^2
(exp4_1 convention).

Selection sub-experiment (near-boundary heuristic under misspecification):
  Candidate library of MISSPECIFIED maps only (exact map deliberately absent):
  omega' in {0.85, 0.95, 1.05, 1.25, 1.60}. Per trial: fit OLS on a degree-2
  polynomial expansion of phi_omega' features (noisy y), rank candidates by
    (i)  5-fold CV on the training block          (paper's negative control)
    (ii) near-boundary validation band x in [2pi, 2pi(1+delta)), delta=0.1
  and compare against the oracle ranking by true OOD error on the full k=1
  block. Reports P(top-1 = oracle top-1), mean Spearman, and whether the
  heuristic picks the least-misspecified available candidate.

Usage
-----
  python rebuttal_e3_misspec_robustness.py --save         # full (3 seeds + selection)
  python rebuttal_e3_misspec_robustness.py --quick        # smoke
  python rebuttal_e3_misspec_robustness.py --cpu          # force CPU
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (set_seed, save_result, sample_torus, ood_torus, phi_torus,
                   target_g_int, MLP, train_mlp, evaluate_mlp, rmse, r2_score,
                   fit_ols, predict_ols, init_timing, collect_meta)

# ---- shared config (mirrors exp4_1) ----------------------------------------
N_TRAIN = 4096
N_ID = 2048
N_OOD = 2048
LAM = 0.35
HIDDEN, LAYERS = 256, 5
EPOCHS = 3000

OMEGAS = [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.25, 1.50]   # axis A (d=2)
D_A = 2
D_B = 4                                                            # axis B
NOISE_S = [0.0, 0.05, 0.1, 0.2, 0.4]                               # axis C (d=2)

# selection sub-experiment
SEL_OMEGAS = [0.85, 0.95, 1.05, 1.25, 1.60]     # exact map absent by design
SEL_D = 2
SEL_N_TRAIN = 512
SEL_N_VAL = 256
SEL_N_OOD = 1024
SEL_DELTA = 0.1            # near-boundary band width as fraction of the period
SEL_Y_NOISE = 0.05
SEL_TRIALS = 100


def phi_omega(x: np.ndarray, omega: float) -> np.ndarray:
    """Per-coordinate Fourier map with (mis)specified frequency omega:
    (sin omega x_1, cos omega x_1, …). omega=1 reproduces utils.phi_torus."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, d = x.shape
    out = np.empty((n, 2 * d), dtype=np.float64)
    out[:, 0::2] = np.sin(omega * x)
    out[:, 1::2] = np.cos(omega * x)
    return out


def phi_partial(x: np.ndarray, m: int) -> np.ndarray:
    """phi (exact, omega=1) on the first m coordinates, raw pass-through for the
    rest: feature dim = 2m + (d - m)."""
    x = np.asarray(x)
    if m == 0:
        return x.copy()
    mapped = phi_torus(x[:, :m])
    if m == x.shape[1]:
        return mapped
    return np.hstack([mapped, x[:, m:]])


def _fit_eval(F_tr, y_tr, F_id, y_id, F_ood, y_ood, epochs, device):
    """Train the repo-standard MLP on features and return exp4_1-style metrics."""
    model = MLP(in_dim=F_tr.shape[1], hidden=HIDDEN, layers=LAYERS)
    train_mlp(model, F_tr.astype(np.float32), y_tr.astype(np.float32),
              epochs=epochs, device=device)
    p_id = evaluate_mlp(model, F_id.astype(np.float32))
    p_ood = evaluate_mlp(model, F_ood.astype(np.float32))
    return dict(id_rmse=rmse(p_id, y_id), ood_rmse=rmse(p_ood, y_ood),
                id_r2=r2_score(p_id, y_id), ood_r2=r2_score(p_ood, y_ood))


def _make_torus_data(d, seed):
    rng = np.random.default_rng(seed)
    theta_tr = sample_torus(d, N_TRAIN, rng)
    theta_id = sample_torus(d, N_ID, rng)
    theta_ood = sample_torus(d, N_OOD, rng)
    x_ood = ood_torus(theta_ood, k_block=1)
    return (theta_tr, target_g_int(theta_tr, lam=LAM),
            theta_id, target_g_int(theta_id, lam=LAM),
            x_ood, target_g_int(theta_ood, lam=LAM))


def axis_wrong_omega(seed, epochs, device):
    th_tr, y_tr, th_id, y_id, x_ood, y_ood = _make_torus_data(D_A, seed * 11 + 1)
    out = {}
    for om in OMEGAS:
        out[f"omega_{om:.2f}"] = _fit_eval(
            phi_omega(th_tr, om), y_tr, phi_omega(th_id, om), y_id,
            phi_omega(x_ood, om), y_ood, epochs, device)
        out[f"omega_{om:.2f}"]["omega"] = om
        print(f"    [A] omega={om:.2f}  ood_rmse={out[f'omega_{om:.2f}']['ood_rmse']:.3f}")
    # raw anchor (no phi at all)
    out["raw"] = _fit_eval(th_tr, y_tr, th_id, y_id, x_ood, y_ood, epochs, device)
    print(f"    [A] raw       ood_rmse={out['raw']['ood_rmse']:.3f}")
    return out


def axis_partial_phi(seed, epochs, device):
    th_tr, y_tr, th_id, y_id, x_ood, y_ood = _make_torus_data(D_B, seed * 11 + 2)
    out = {}
    for m in range(D_B + 1):
        out[f"m_{m}"] = _fit_eval(
            phi_partial(th_tr, m), y_tr, phi_partial(th_id, m), y_id,
            phi_partial(x_ood, m), y_ood, epochs, device)
        out[f"m_{m}"]["m"] = m
        print(f"    [B] m={m}/{D_B}  ood_rmse={out[f'm_{m}']['ood_rmse']:.3f}")
    return out


def axis_noisy_phi(seed, epochs, device):
    th_tr, y_tr, th_id, y_id, x_ood, y_ood = _make_torus_data(D_A, seed * 11 + 3)
    rng = np.random.default_rng(seed * 11 + 4)
    out = {}
    for s in NOISE_S:
        def noisy(x):
            F = phi_torus(x)
            return F + rng.normal(0.0, s, size=F.shape) if s > 0 else F
        out[f"s_{s:.2f}"] = _fit_eval(
            noisy(th_tr), y_tr, noisy(th_id), y_id, noisy(x_ood), y_ood,
            epochs, device)
        out[f"s_{s:.2f}"]["s"] = s
        print(f"    [C] s={s:.2f}  ood_rmse={out[f's_{s:.2f}']['ood_rmse']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Selection sub-experiment (OLS on degree-2 expansion of candidate features)
# ---------------------------------------------------------------------------

def _poly2(F: np.ndarray) -> np.ndarray:
    """[F, F_i·F_j for i<=j] (bias added inside fit_ols/predict_ols)."""
    n, k = F.shape
    cols = [F]
    prods = [F[:, i:i + 1] * F[:, j:j + 1] for i in range(k) for j in range(i, k)]
    return np.hstack(cols + prods)


def _sel_features(x, omega):
    return _poly2(phi_omega(x, omega))


def selection_trial(trial_seed):
    rng = np.random.default_rng(trial_seed)
    theta_tr = sample_torus(SEL_D, SEL_N_TRAIN, rng)
    y_tr = target_g_int(theta_tr, lam=LAM) + rng.normal(0, SEL_Y_NOISE, SEL_N_TRAIN)

    # near-boundary band: x just beyond the training block, x in [2pi, 2pi(1+delta))
    theta_v = rng.uniform(0.0, 2.0 * np.pi * SEL_DELTA, size=(SEL_N_VAL, SEL_D))
    x_val = theta_v + 2.0 * np.pi
    y_val = target_g_int(theta_v, lam=LAM) + rng.normal(0, SEL_Y_NOISE, SEL_N_VAL)

    theta_o = sample_torus(SEL_D, SEL_N_OOD, rng)
    x_ood = ood_torus(theta_o, k_block=1)
    y_ood = target_g_int(theta_o, lam=LAM)          # noiseless oracle ranking

    cv_sc, val_sc, oracle_sc = {}, {}, {}
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for om in SEL_OMEGAS:
        name = f"{om:.2f}"
        F_tr = _sel_features(theta_tr, om)
        w = fit_ols(F_tr, y_tr)
        val_sc[name] = rmse(predict_ols(_sel_features(x_val, om), w), y_val)
        oracle_sc[name] = rmse(predict_ols(_sel_features(x_ood, om), w), y_ood)
        errs = []
        for tr_i, va_i in kf.split(theta_tr):
            wf = fit_ols(F_tr[tr_i], y_tr[tr_i])
            errs.append(rmse(predict_ols(F_tr[va_i], wf), y_tr[va_i]))
        cv_sc[name] = float(np.mean(errs))

    names = [f"{om:.2f}" for om in SEL_OMEGAS]
    rank = lambda sc: np.argsort(np.argsort([sc[n] for n in names]))
    return dict(
        cv_sel=min(cv_sc, key=cv_sc.get),
        val_sel=min(val_sc, key=val_sc.get),
        oracle_sel=min(oracle_sc, key=oracle_sc.get),
        spearman_val_oracle=float(spearmanr(rank(val_sc), rank(oracle_sc)).statistic),
        spearman_cv_oracle=float(spearmanr(rank(cv_sc), rank(oracle_sc)).statistic),
        cv_scores=cv_sc, val_scores=val_sc, oracle_scores=oracle_sc,
    )


def run_selection(base_seed, n_trials, save):
    t0 = init_timing()
    names = [f"{om:.2f}" for om in SEL_OMEGAS]
    least_misspec = {f"{om:.2f}" for om in SEL_OMEGAS
                     if abs(om - 1.0) == min(abs(o - 1.0) for o in SEL_OMEGAS)}
    counts = {k: {n: 0 for n in names} for k in ("cv", "val", "oracle")}
    cv_hits = val_hits = val_in_least = 0
    sp_val, sp_cv = [], []
    detail = None
    for t in range(n_trials):
        r = selection_trial(base_seed * 100000 + t)
        counts["cv"][r["cv_sel"]] += 1
        counts["val"][r["val_sel"]] += 1
        counts["oracle"][r["oracle_sel"]] += 1
        cv_hits += (r["cv_sel"] == r["oracle_sel"])
        val_hits += (r["val_sel"] == r["oracle_sel"])
        val_in_least += (r["val_sel"] in least_misspec)
        sp_val.append(r["spearman_val_oracle"])
        sp_cv.append(r["spearman_cv_oracle"])
        if t == 0:
            detail = r
    out = dict(
        experiment_id="E3_misspec_selection",
        name="near_boundary_selection_under_misspecification",
        seed=base_seed,
        config=dict(candidate_omegas=SEL_OMEGAS, exact_map_in_library=False,
                    d=SEL_D, n_train=SEL_N_TRAIN, n_val=SEL_N_VAL, n_ood=SEL_N_OOD,
                    delta_band=SEL_DELTA, y_noise=SEL_Y_NOISE, n_trials=n_trials,
                    model="OLS on degree-2 expansion of phi_omega features",
                    least_misspecified=sorted(least_misspec)),
        results=dict(
            val_top1_matches_oracle=val_hits / n_trials,
            cv_top1_matches_oracle=cv_hits / n_trials,
            val_top1_in_least_misspecified=val_in_least / n_trials,
            random_baseline=1.0 / len(SEL_OMEGAS),
            spearman_val_oracle_mean=float(np.mean(sp_val)),
            spearman_cv_oracle_mean=float(np.mean(sp_cv)),
            counts=counts,
            trial0_scores=dict(cv=detail["cv_scores"], val=detail["val_scores"],
                               oracle=detail["oracle_scores"]),
        ),
    )
    out.update(collect_meta(t0))
    print("\n  [selection] over", n_trials, "trials:")
    print(f"    P(val top-1 = oracle top-1) = {out['results']['val_top1_matches_oracle']:.2f}"
          f"   (CV: {out['results']['cv_top1_matches_oracle']:.2f},"
          f" random: {1.0/len(SEL_OMEGAS):.2f})")
    print(f"    P(val top-1 in least-misspecified {sorted(least_misspec)}) ="
          f" {out['results']['val_top1_in_least_misspecified']:.2f}")
    print(f"    Spearman(val, oracle) = {np.mean(sp_val):.3f}"
          f"   Spearman(cv, oracle) = {np.mean(sp_cv):.3f}")
    print(f"    counts: {counts}")
    if save:
        save_result(out)
    return out


def run(seed, epochs, device, save):
    set_seed(seed)
    t0 = init_timing()
    print(f"  [seed {seed}] axis A (wrong omega, d={D_A}) …")
    ax_a = axis_wrong_omega(seed, epochs, device)
    print(f"  [seed {seed}] axis B (partial phi, d={D_B}) …")
    ax_b = axis_partial_phi(seed, epochs, device)
    print(f"  [seed {seed}] axis C (noisy phi, d={D_A}) …")
    ax_c = axis_noisy_phi(seed, epochs, device)
    out = dict(
        experiment_id="E3_misspec",
        name="phi_misspecification_robustness_torus",
        seed=seed,
        config=dict(n_train=N_TRAIN, n_id=N_ID, n_ood=N_OOD, lam=LAM,
                    hidden=HIDDEN, layers=LAYERS, epochs=epochs,
                    d_axis_a=D_A, d_axis_b=D_B, omegas=OMEGAS, noise_s=NOISE_S,
                    ood_construction="x = theta + 2pi (k=1 block)"),
        axes=dict(wrong_omega=ax_a, partial_phi=ax_b, noisy_phi=ax_c),
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


def aggregate_print(runs):
    print("\n=== E3 aggregate (mean±std over seeds) — OOD RMSE ===")
    print("\n  Axis A — wrong frequency (d=2, target std ≈ 0.47):")
    keys = [f"omega_{om:.2f}" for om in OMEGAS] + ["raw"]
    for k in keys:
        v = [r["axes"]["wrong_omega"][k]["ood_rmse"] for r in runs]
        i = [r["axes"]["wrong_omega"][k]["id_rmse"] for r in runs]
        print(f"    {k:<12} id={np.mean(i):.3f}±{np.std(i):.3f}"
              f"   ood={np.mean(v):.3f}±{np.std(v):.3f}")
    print("\n  Axis B — partial phi (d=4, m of 4 coordinates mapped):")
    for m in range(D_B + 1):
        v = [r["axes"]["partial_phi"][f"m_{m}"]["ood_rmse"] for r in runs]
        print(f"    m={m}   ood={np.mean(v):.3f}±{np.std(v):.3f}")
    print("\n  Axis C — noisy phi (d=2, feature noise s):")
    for s in NOISE_S:
        v = [r["axes"]["noisy_phi"][f"s_{s:.2f}"]["ood_rmse"] for r in runs]
        print(f"    s={s:.2f}  ood={np.mean(v):.3f}±{np.std(v):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--n-trials", type=int, default=SEL_TRIALS)
    p.add_argument("--quick", action="store_true", help="smoke: 1 seed, 300 epochs, 10 trials")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--skip-selection", action="store_true")
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 300
        args.n_trials = 10

    device = torch.device("cpu") if args.cpu else None
    if device is None:
        from utils import DEVICE as device

    print("=== Rebuttal E3 — phi misspecification robustness (torus) ===")
    print(f"  device={device}  seeds={args.seeds}  epochs={args.epochs}")

    runs = [run(s, args.epochs, device, save=args.save) for s in args.seeds]
    aggregate_print(runs)
    if not args.skip_selection:
        run_selection(base_seed=args.seeds[0] + 7, n_trials=args.n_trials,
                      save=args.save)
