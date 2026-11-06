"""
Exp 2.1 regime sweep — selection accuracy and structural-commitment OOD vs noise σ
==================================================================================
Sweeps Gaussian noise σ across signal- and noise-dominated regimes of Prop 2.2,
keeping the rest of Exp 2.1 fixed (DGP sin(x), library {fourier, poly7, poly9, raw},
n_train=200, 100 trials, near-boundary [0, π], OOD [π, 3π]).

For each σ we record:
  - CV selection accuracy  (fraction of trials where 5-fold CV picks fourier_ols)
  - Near-boundary selection accuracy  (fraction picking fourier_ols)
  - Fourier-OLS OOD error  (the structural commitment, regime-invariant by hypothesis)

The Le Cam regime indicator is ε√n/σ where ε = sup|sin - Tay9| on [-π, 0] ≈ 6.93e-3.
At fixed n=200, ε√n ≈ 0.098, so the regime crossover is around σ ≈ 0.1.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, save_result, init_timing, collect_meta, ood_relative_error
from exp3_1_selection_protocol import (
    LIBRARY, CORRECT, RANDOM_BASE, f_true, make_features,
    ols_fit, ols_predict, cv_score,
    X_TRAIN_LO, X_TRAIN_HI, X_BNDRY_HI, X_OOD_HI,
)


SIGMAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
N_TRAIN = 200
N_TRIALS = 100

# ε for the closest competitor (Tay9) on the training window — used to label the
# regime boundary in the output JSON. Numerical max of |sin(x) - Tay9(x)| on
# [-π, 0] (≈ 6.93e-3); the OLS library competes against poly7 and poly9, so the
# *worst* competitor has ε ≈ 7.5e-2 (Tay7) but Tay9 is the binding constraint
# for the canonical-vs-closest-confound argument.
EPS_TAY9 = 6.925e-3
EPS_TAY7 = 7.522e-2


def run_trial_noisy_val(trial_seed: int, noise_std: float, n_train: int):
    """Like exp3_1_selection_protocol.run_trial, but the near-boundary validation
    targets are also corrupted with N(0, σ²) noise so CV and near-boundary
    operate on data with the same noise level (Le Cam's apples-to-apples setup).
    OOD targets remain clean (oracle reference for the structural-commitment
    OOD-error curve). Returns (cv_sel, val_sel, oracle_scores)."""
    rng = np.random.default_rng(trial_seed)

    x_train = np.linspace(X_TRAIN_LO, X_TRAIN_HI, n_train).reshape(-1, 1)
    x_val   = np.linspace(X_TRAIN_HI, X_BNDRY_HI, n_train // 2).reshape(-1, 1)
    x_ood   = np.linspace(X_BNDRY_HI, X_OOD_HI,   n_train).reshape(-1, 1)

    y_train = f_true(x_train) + rng.normal(0, noise_std, n_train)
    y_val   = f_true(x_val)   + rng.normal(0, noise_std, n_train // 2)
    y_ood   = f_true(x_ood)

    cv_scores = {}
    val_scores = {}
    oracle_scores = {}
    for name in LIBRARY:
        X_tr = make_features(name, x_train)
        X_vl = make_features(name, x_val)
        X_oo = make_features(name, x_ood)
        w = ols_fit(X_tr, y_train)
        cv_scores[name]     = cv_score(name, x_train, y_train)
        val_scores[name]    = ood_relative_error(ols_predict(X_vl, w), y_val)
        oracle_scores[name] = ood_relative_error(ols_predict(X_oo, w), y_ood)

    cv_sel  = min(cv_scores,  key=cv_scores.get)
    val_sel = min(val_scores, key=val_scores.get)
    return cv_sel, val_sel, oracle_scores


def run_one_sigma(seed: int, noise_std: float, n_trials: int, n_train: int) -> dict:
    """Run n_trials at one σ; track CV / near-boundary selection accuracy AND the
    per-condition OOD error trajectory (Fourier vs poly7 vs poly9 vs raw).
    The per-condition OOD scaling is the structural-commitment story: which
    candidates have bounded OOD vs which catastrophically extrapolate."""
    cv_correct = 0
    val_correct = 0
    cv_counts = {t: 0 for t in LIBRARY}
    val_counts = {t: 0 for t in LIBRARY}
    ood_errors_per_cond = {t: [] for t in LIBRARY}

    for trial in range(n_trials):
        trial_seed = seed * 10000 + trial
        cv_sel, val_sel, oracle_sc = run_trial_noisy_val(
            trial_seed, noise_std, n_train)

        cv_counts[cv_sel] += 1
        val_counts[val_sel] += 1
        if cv_sel == CORRECT:
            cv_correct += 1
        if val_sel == CORRECT:
            val_correct += 1
        for name in LIBRARY:
            ood_errors_per_cond[name].append(oracle_sc[name])

    ood_summary = {
        name: dict(
            mean=float(np.mean(ood_errors_per_cond[name])),
            median=float(np.median(ood_errors_per_cond[name])),
            std=float(np.std(ood_errors_per_cond[name])),
        )
        for name in LIBRARY
    }
    return dict(
        noise_std=float(noise_std),
        cv_accuracy=cv_correct / n_trials,
        val_accuracy=val_correct / n_trials,
        cv_counts=dict(cv_counts),
        val_counts=dict(val_counts),
        ood_per_condition=ood_summary,
        regime_ratio_tay9=float(EPS_TAY9 * np.sqrt(n_train) / noise_std),
        regime_ratio_tay7=float(EPS_TAY7 * np.sqrt(n_train) / noise_std),
    )


def run(seed: int = 42, n_trials: int = N_TRIALS, n_train: int = N_TRAIN,
        sigmas=SIGMAS, save: bool = False) -> dict:
    set_seed(seed)
    t0 = init_timing()

    print(f"\n=== Exp 2.1 Regime Sweep ===")
    print(f"  Library: {LIBRARY}  (correct: {CORRECT}, K={len(LIBRARY)}, "
          f"random baseline={RANDOM_BASE:.2f})")
    print(f"  Training: x ∈ [{X_TRAIN_LO:.2f}, {X_TRAIN_HI:.2f}]  "
          f"(n_train={n_train})")
    print(f"  Near-boundary val: x ∈ [{X_TRAIN_HI:.2f}, {X_BNDRY_HI:.2f}]  "
          f"OOD: x ∈ [{X_BNDRY_HI:.2f}, {X_OOD_HI:.2f}]")
    print(f"  σ sweep: {list(sigmas)}  ({n_trials} trials each)\n")

    per_sigma = []
    for sigma in sigmas:
        r = run_one_sigma(seed, sigma, n_trials, n_train)
        per_sigma.append(r)
        ood = r["ood_per_condition"]
        print(f"  σ={sigma:>7.4f}  ratio(Tay9)={r['regime_ratio_tay9']:>7.3f}  "
              f"CV={r['cv_accuracy']:.2f}  NB={r['val_accuracy']:.2f}  "
              f"OOD-median: fourier={ood['fourier_ols']['median']:7.2f}  "
              f"poly9={ood['poly9_ols']['median']:.1e}  "
              f"raw={ood['raw_ols']['median']:5.1f}")

    out = dict(
        experiment_id="3.1_regime_sweep",
        name="selection_protocol_regime_sweep",
        seed=seed,
        n_trials=n_trials,
        config=dict(
            n_train=n_train,
            sigmas=list(sigmas),
            K=len(LIBRARY),
            random_baseline=RANDOM_BASE,
            eps_tay9=EPS_TAY9,
            eps_tay7=EPS_TAY7,
            train_window=[X_TRAIN_LO, X_TRAIN_HI],
            near_boundary_window=[X_TRAIN_HI, X_BNDRY_HI],
            ood_window=[X_BNDRY_HI, X_OOD_HI],
        ),
        sweep=per_sigma,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, n_trials=args.n_trials, n_train=args.n_train,
        save=args.save)
