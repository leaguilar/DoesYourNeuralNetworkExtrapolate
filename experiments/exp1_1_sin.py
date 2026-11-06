"""
Exp 1.1 — Observationally equivalent DGPs: sin(x) vs. Taylor polynomial
=========================================================================
P1: f(x) = sin(x)                            (true DGP — periodic)
P2: f(x) = x - x³/6 + x⁵/120 - x⁷/5040 + x⁹/362880   (Taylor deg-9)

On [-π, π]:  max |P1-P2| / |P1| < 1%  → statistically indistinguishable.
On [π, 3π]:  divergence can exceed 10⁴.

Conditions: raw_mlp, fourier_mlp, fourier_ols, taylor9_direct (regression);
            tabpfn, tabpfn_fourier (zero-shot tabular FM);
            timesfm (zero-shot time-series FM).

Use --skip-foundation to run only MLP/OLS conditions (no TabPFN / TimesFM).
"""

import argparse
import math
import os
import sys

import numpy as np
import pysindy as ps

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_seed, save_result, ood_relative_error,
                   spectral_complexity, MLP, train_mlp, evaluate_mlp,
                   init_timing, collect_meta)
from models import TabPFNModel, TimesFMModel, SINDyModel  # SINDyModel kept for compatibility; SINDy results now go through ps.PolynomialLibrary below.


N_TRAIN = 128
N_OOD   = 128


def p1_sin(x):    return np.sin(x.flatten())
def p2_taylor9(x):
    xf = x.flatten()
    return xf - xf**3/6 + xf**5/120 - xf**7/5040 + xf**9/362880

def fourier(x):
    return np.concatenate([np.sin(x), np.cos(x)], axis=1)


def run(seed: int = 42, epochs: int = 3000,
        backend: str = "cpu", skip_foundation: bool = False,
        save: bool = False):
    set_seed(seed)
    t0 = init_timing()

    step    = 2 * np.pi / (N_TRAIN - 1)
    x_all   = -np.pi + step * np.arange(N_TRAIN + N_OOD)
    x_train = x_all[:N_TRAIN].reshape(-1, 1)
    x_ood   = x_all[N_TRAIN:].reshape(-1, 1)
    x_id    = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)

    y_train    = p1_sin(x_train)
    y_train_p2 = p2_taylor9(x_train)
    y_id       = p1_sin(x_id)
    y_id_p2    = p2_taylor9(x_id)
    y_ood      = p1_sin(x_ood)
    y_ood_p2   = p2_taylor9(x_ood)

    train_diff = np.abs(y_train - y_train_p2)
    ood_diff   = np.abs(y_ood   - y_ood_p2)
    y_scale    = np.abs(y_train).mean() + 1e-12

    print("\n=== Exp 1.1  Observational Equivalence: sin(x) vs. Taylor₉ ===")
    print(f"\n  Grid: step={step:.4f}  train=[{x_train[0,0]:.3f},{x_train[-1,0]:.3f}]"
          f"  OOD=[{x_ood[0,0]:.3f},{x_ood[-1,0]:.3f}]")
    print(f"\n  Indistinguishability on [-π, π]:")
    print(f"    max |P1-P2| = {train_diff.max():.4f}  "
          f"(relative: {train_diff.max()/y_scale*100:.2f}%)")
    print(f"  OOD divergence on [π, ≈3π]:")
    print(f"    max |P1-P2| = {ood_diff.max():.2f}")

    results = {}

    # ── raw MLP ─────────────────────────────────────────────────────────────
    m = MLP(1)
    train_mlp(m, x_train, y_train, epochs=epochs)
    y_raw_ood = evaluate_mlp(m, x_ood)
    results["raw_mlp"] = dict(
        id_error        = ood_relative_error(evaluate_mlp(m, x_id), y_id),
        ood_error       = ood_relative_error(y_raw_ood, y_ood),
        ood_error_vs_p2 = ood_relative_error(y_raw_ood, y_ood_p2),
        sc_train        = spectral_complexity(x_train, y_train),
        sc_ood          = spectral_complexity(x_ood,   y_ood),
    )
    m_p2 = MLP(1)
    train_mlp(m_p2, x_train, y_train_p2, epochs=epochs)
    y_raw_p2_ood = evaluate_mlp(m_p2, x_ood)
    results["raw_mlp_p2ctx"] = dict(
        id_error        = ood_relative_error(evaluate_mlp(m_p2, x_id), y_id_p2),
        ood_error       = ood_relative_error(y_raw_p2_ood, y_ood_p2),
        ood_error_vs_p1 = ood_relative_error(y_raw_p2_ood, y_ood),
        sc_train=float("nan"), sc_ood=float("nan"),
    )

    # ── Fourier MLP ──────────────────────────────────────────────────────────
    m = MLP(2)
    train_mlp(m, fourier(x_train), y_train, epochs=epochs)
    y_fmlp_ood = evaluate_mlp(m, fourier(x_ood))
    results["fourier_mlp"] = dict(
        id_error        = ood_relative_error(evaluate_mlp(m, fourier(x_id)), y_id),
        ood_error       = ood_relative_error(y_fmlp_ood, y_ood),
        ood_error_vs_p2 = ood_relative_error(y_fmlp_ood, y_ood_p2),
        sc_train        = spectral_complexity(fourier(x_train), y_train),
        sc_ood          = spectral_complexity(fourier(x_ood),   y_ood),
    )

    # ── Fourier OLS ──────────────────────────────────────────────────────────
    Xff  = np.hstack([fourier(x_train), np.ones((N_TRAIN, 1))])
    w_ff, _, _, _ = np.linalg.lstsq(Xff, y_train, rcond=None)
    def _pred_fols(x): return np.hstack([fourier(x), np.ones((len(x), 1))]) @ w_ff
    y_fols_ood = _pred_fols(x_ood).flatten()
    results["fourier_ols"] = dict(
        id_error        = ood_relative_error(_pred_fols(x_id).flatten(), y_id),
        ood_error       = ood_relative_error(y_fols_ood, y_ood),
        ood_error_vs_p2 = ood_relative_error(y_fols_ood, y_ood_p2),
        sc_train        = spectral_complexity(fourier(x_train), y_train),
        sc_ood          = spectral_complexity(fourier(x_ood),   y_ood),
    )

    # ── Taylor-9 direct ──────────────────────────────────────────────────────
    y_t9_ood = p2_taylor9(x_ood)
    results["taylor9_direct"] = dict(
        id_error        = ood_relative_error(p2_taylor9(x_id), y_id),
        ood_error       = ood_relative_error(y_t9_ood, y_ood),
        ood_error_vs_p2 = ood_relative_error(y_t9_ood, y_ood_p2),
        sc_train=float("nan"), sc_ood=float("nan"),
    )

    # ── SINDy ─────────────────────────────────────────────────────────────────
    # All four arms use the same library: ps.PolynomialLibrary(degree=9,
    # include_bias=True), matching exp1_2, exp1_3, exp2_1, exp2_2. Only the
    # input Z varies — raw x or the Fourier feature map (sin x, cos x).

    def _sindy_polylib(Z_tr, y_tr, Z_id, Z_ood,
                       degree: int = 9, threshold: float = 1e-6):
        """Sparse poly regression via pysindy with ps.PolynomialLibrary.

        Mirrors exp2_2_kepler:_sindy_static / exp2_1_mak_vf:_sindy_delta.
        Returns predictions on (id, ood, tr) plus the squared-NMSE residual
        delta on train/ood and the active-term count.
        """
        lib = ps.PolynomialLibrary(degree=degree, include_bias=True)
        lib.fit(Z_tr)
        Th_tr = np.asarray(lib.transform(Z_tr))
        Th_id = np.asarray(lib.transform(Z_id))
        Th_oo = np.asarray(lib.transform(Z_ood))

        opt = ps.STLSQ(threshold=threshold, alpha=0.0, max_iter=50)
        opt.fit(Th_tr, np.asarray(y_tr).reshape(-1, 1))
        xi = np.asarray(opt.coef_[0])

        pred_tr  = Th_tr @ xi
        pred_id  = Th_id @ xi
        pred_ood = Th_oo @ xi
        return pred_tr, pred_id, pred_ood, xi

    def _delta_pair(y_tr, pred_tr, y_ood, pred_ood):
        dtr  = float(np.sum((y_tr  - pred_tr ) ** 2) / (np.sum(y_tr  ** 2) + 1e-12))
        dood = float(np.sum((y_ood - pred_ood) ** 2) / (np.sum(y_ood ** 2) + 1e-12))
        return dtr, dood

    Z_raw_tr,  Z_raw_id,  Z_raw_ood  = x_train, x_id, x_ood
    Z_four_tr, Z_four_id, Z_four_ood = fourier(x_train), fourier(x_id), fourier(x_ood)

    # SINDy: Fourier features + PolynomialLibrary(deg=9), P1 data
    pred_tr, pred_id, pred_ood, xi = _sindy_polylib(
        Z_four_tr, y_train, Z_four_id, Z_four_ood)
    dtr, dood = _delta_pair(y_train, pred_tr, y_ood, pred_ood)
    results["sindy_fourier"] = dict(
        id_error        = ood_relative_error(pred_id,  y_id),
        ood_error       = ood_relative_error(pred_ood, y_ood),
        ood_error_vs_p2 = ood_relative_error(pred_ood, y_ood_p2),
        n_terms         = int((np.abs(xi) > 1e-10).sum()),
        delta_train     = dtr,
        delta_ood       = dood,
    )

    # SINDy: raw x + PolynomialLibrary(deg=9), P1 data — ID-equivalent to Taylor expansion
    pred_tr, pred_id, pred_ood, xi = _sindy_polylib(
        Z_raw_tr, y_train, Z_raw_id, Z_raw_ood)
    dtr, dood = _delta_pair(y_train, pred_tr, y_ood, pred_ood)
    results["sindy_raw"] = dict(
        id_error        = ood_relative_error(pred_id,  y_id),
        ood_error       = ood_relative_error(pred_ood, y_ood),
        ood_error_vs_p2 = ood_relative_error(pred_ood, y_ood_p2),
        n_terms         = int((np.abs(xi) > 1e-10).sum()),
        delta_train     = dtr,
        delta_ood       = dood,
    )

    # SINDy: raw x + PolynomialLibrary(deg=9), P2 data — recovers Taylor-9 exactly
    pred_tr, pred_id, pred_ood, xi = _sindy_polylib(
        Z_raw_tr, y_train_p2, Z_raw_id, Z_raw_ood)
    dtr, dood = _delta_pair(y_train_p2, pred_tr, y_ood_p2, pred_ood)
    results["sindy_p2ctx"] = dict(
        id_error        = ood_relative_error(pred_id,  y_id_p2),
        ood_error       = ood_relative_error(pred_ood, y_ood_p2),
        ood_error_vs_p1 = ood_relative_error(pred_ood, y_ood),
        n_terms         = int((np.abs(xi) > 1e-10).sum()),
        delta_train     = dtr,
        delta_ood       = dood,
    )

    # SINDy: Fourier features + PolynomialLibrary(deg=9), P2 data — commits to sin(x)
    pred_tr, pred_id, pred_ood, xi = _sindy_polylib(
        Z_four_tr, y_train_p2, Z_four_id, Z_four_ood)
    dtr, dood = _delta_pair(y_train_p2, pred_tr, y_ood_p2, pred_ood)
    results["sindy_fourier_p2ctx"] = dict(
        id_error        = ood_relative_error(pred_id,  y_id_p2),
        ood_error       = ood_relative_error(pred_ood, y_ood_p2),
        ood_error_vs_p1 = ood_relative_error(pred_ood, y_ood),
        n_terms         = int((np.abs(xi) > 1e-10).sum()),
        delta_train     = dtr,
        delta_ood       = dood,
    )

    if not skip_foundation:
        # ── TabPFN ───────────────────────────────────────────────────────────
        tpfn = TabPFNModel()
        if tpfn.available:
            print("\n  Loading TabPFN (cloud inference) ...")
            tpfn.fit(x_train, y_train)
            y_tpfn_ood = tpfn.predict(x_ood)
            results["tabpfn"] = dict(
                id_error        = ood_relative_error(tpfn.predict(x_id), y_id),
                ood_error       = ood_relative_error(y_tpfn_ood, y_ood),
                ood_error_vs_p2 = ood_relative_error(y_tpfn_ood, y_ood_p2),
                sc_train=float("nan"), sc_ood=float("nan"),
            )
            # tabpfn P2 context: fed Taylor-9 training data
            tpfn_p2 = TabPFNModel()
            tpfn_p2.fit(x_train, y_train_p2)
            y_p2ctx_ood = tpfn_p2.predict(x_ood)
            results["tabpfn_p2ctx"] = dict(
                id_error        = float("nan"),
                ood_error       = ood_relative_error(y_p2ctx_ood, y_ood_p2),
                ood_error_vs_p1 = ood_relative_error(y_p2ctx_ood, y_ood),
                sc_train=float("nan"), sc_ood=float("nan"),
            )
            # tabpfn Fourier features
            tpfn_ff = TabPFNModel()
            tpfn_ff.fit(fourier(x_train), y_train)
            y_tpfn_ff_ood = tpfn_ff.predict(fourier(x_ood))
            results["tabpfn_fourier"] = dict(
                id_error        = ood_relative_error(tpfn_ff.predict(fourier(x_id)), y_id),
                ood_error       = ood_relative_error(y_tpfn_ff_ood, y_ood),
                ood_error_vs_p2 = ood_relative_error(y_tpfn_ff_ood, y_ood_p2),
                sc_train=float("nan"), sc_ood=float("nan"),
            )
        else:
            print("\n  TabPFN not available (skip)")

        # ── TimesFM ──────────────────────────────────────────────────────────
        print("\n  Loading TimesFM 1.0-200M ...")
        tfm = TimesFMModel(context_len=N_TRAIN, horizon_len=N_OOD, backend=backend)
        tfm.set_context(y_train.flatten())
        y_fore = tfm.forecast(N_OOD)
        results["timesfm"] = dict(
            id_error        = None,
            ood_error       = ood_relative_error(y_fore, y_ood.flatten()),
            ood_error_vs_p2 = ood_relative_error(y_fore, y_ood_p2.flatten()),
            sc_train=float("nan"), sc_ood=float("nan"),
        )
        # timesfm Fourier covariates, P1 context
        x_all_cov = np.concatenate([x_train.flatten(), x_ood.flatten()])
        cov_f1 = {
            "sin_x": np.sin(x_all_cov).astype(np.float32),
            "cos_x": np.cos(x_all_cov).astype(np.float32),
        }
        y_fore_fc = tfm.forecast_with_covariates(
            y_train.flatten().astype(np.float32), cov_f1, N_OOD
        )
        results["timesfm_fourier_cov"] = dict(
            id_error        = None,
            ood_error       = ood_relative_error(y_fore_fc, y_ood.flatten()),
            ood_error_vs_p2 = ood_relative_error(y_fore_fc, y_ood_p2.flatten()),
        )
        # timesfm P2 context
        tfm.set_context(y_train_p2.flatten())
        y_fore_p2 = tfm.forecast(N_OOD)
        results["timesfm_p2ctx"] = dict(
            id_error        = None,
            ood_error       = ood_relative_error(y_fore_p2, y_ood_p2),
            ood_error_vs_p1 = ood_relative_error(y_fore_p2, y_ood),
            sc_train=float("nan"), sc_ood=float("nan"),
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  {'condition':<22}  {'ID err%':>8}  {'OOD err%':>9}")
    for k, v in results.items():
        ie = v.get("id_error")
        id_s = "--" if ie is None else ("n/a" if (isinstance(ie, float) and math.isnan(ie)) else f"{ie:>8.2f}")
        print(f"  {k:<22}  {id_s:>8}  {v['ood_error']:>9.2f}")

    out = dict(
        experiment_id = "1.1",
        name          = "sin_observational_equivalence",
        seed          = seed,
        epochs        = epochs,
        n_train       = N_TRAIN,
        n_ood         = N_OOD,
        indistinguishability = dict(
            train_max_abs_diff     = float(train_diff.max()),
            train_max_rel_diff_pct = float(train_diff.max() / y_scale * 100),
            ood_max_abs_diff       = float(ood_diff.max()),
        ),
        conditions = {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",             type=int,  default=42)
    parser.add_argument("--epochs",           type=int,  default=3000)
    parser.add_argument("--backend",          type=str,  default="cpu",
                        choices=["cpu", "gpu"])
    parser.add_argument("--skip-foundation",  action="store_true",
                        help="Skip TabPFN and TimesFM (faster local run)")
    parser.add_argument("--save",             action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, backend=args.backend,
        skip_foundation=args.skip_foundation, save=args.save)
