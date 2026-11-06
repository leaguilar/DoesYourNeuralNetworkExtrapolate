"""
Exp 1.2 — DGP specificity: power law vs. exponential disambiguation
====================================================================
DGPs calibrated to agree at both endpoints of training window [1, 2]:
  P1: f(x) = x²                      (power law;   log-log → linear)
  P2: f(x) = exp(α(x-1))             (exponential; log-y  → linear)
  α = 2·log(2) ≈ 1.386   so that P1(2) = P2(2) = 4

With noise (std ≈ 0.2), indistinguishable on [1, 2]; wildly divergent on [2, 10].

Regression conditions (per DGP): raw_mlp, log_log_ols, log_y_ols,
                                  log_log_mlp, log_y_mlp,
                                  sindy_loglog, sindy_logy, sindy_raw,
                                  tabpfn (raw / loglog / logy),
                                  timesfm (raw / logy / loglog).

TimesFM feature-map conditions mirror the OLS pairing:
  - timesfm        : forecast y          on raw context  (no feature map)
  - timesfm_logy   : forecast log y      with x      as dynamic covariate
                     (target transform + raw x feature; correct for P2)
  - timesfm_loglog : forecast log y      with log(x) as dynamic covariate
                     (target transform + log x feature; correct for P1)
TimesFM context covers [1,2] with 128 points and the forecast horizon spans
the full [2,10] OOD window (≈1016 autoregressive steps, 8 patches of 128) so
its OOD numbers are evaluated on the same window as OLS / SINDy / TabPFN.

Use --skip-foundation to run only the regression / SINDy conditions
(no TabPFN / TimesFM).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_seed, save_result, ood_relative_error,
                   spectral_complexity, MLP, train_mlp, evaluate_mlp,
                   init_timing, collect_meta)
from models import TabPFNModel, TimesFMModel


X_LO, X_HI = 1.0, 2.0
X_OOD       = 10.0
B           = 2.0
ALPHA = B * np.log(X_HI / X_LO) / (X_HI - X_LO)   # ≈ 1.386

# TimesFM context / forecast grid.
# Context covers [X_LO, X_HI] = [1, 2] uniformly (128 points, step ≈ 1/127).
# Forecast horizon spans the FULL OOD window [X_HI, X_OOD] = [2, 10] at the
# same step, matching the OLS / SINDy / TabPFN evaluation window. The model
# autoregresses over N_OOD_TFM // output_patch_len patches.
N_CTX_TFM     = 128
STEP_TFM      = (X_HI - X_LO) / (N_CTX_TFM - 1)
N_OOD_TFM     = int(np.ceil((X_OOD - X_HI) / STEP_TFM))   # ≈ 1016
HORIZON_TFM   = ((N_OOD_TFM + 127) // 128) * 128          # round up to patch (1024)


def dgp_power(x): return np.power(x.flatten(), B)
def dgp_exp(x):   return np.exp(ALPHA * (x.flatten() - X_LO))


def _cross_ood(pred_ood, x_ood):
    """Same OOD predictions evaluated against both P1 and P2 targets."""
    pred = np.asarray(pred_ood).flatten()
    return (
        ood_relative_error(pred, dgp_power(x_ood)),
        ood_relative_error(pred, dgp_exp(x_ood)),
    )


def run_one_dgp(dgp_fn, dgp_name, seed, epochs, noise_std=0.2,
                n_train=150, n_test=300):
    """Regression conditions on a noisy training grid."""
    rng = np.random.default_rng(seed)

    x_train = np.linspace(X_LO, X_HI,  n_train).reshape(-1, 1)
    x_id    = np.linspace(X_LO, X_HI,  n_test).reshape(-1, 1)
    x_ood   = np.linspace(X_HI, X_OOD, n_test).reshape(-1, 1)

    y_train_clean = dgp_fn(x_train)
    y_train = y_train_clean + rng.normal(0, noise_std, size=y_train_clean.shape)
    y_id    = dgp_fn(x_id)
    y_ood   = dgp_fn(x_ood)
    ly_tr   = np.log(np.maximum(y_train, 1e-8))

    results = {}

    # (1) Raw MLP
    y_scale = float(np.abs(y_train).max())
    m = MLP(1)
    train_mlp(m, x_train / X_HI, y_train / y_scale, epochs=epochs)
    pred_ood_raw = evaluate_mlp(m, x_ood / X_HI) * y_scale
    e_p1, e_p2 = _cross_ood(pred_ood_raw, x_ood)
    results["raw_mlp"] = dict(
        id_error        = ood_relative_error(evaluate_mlp(m, x_id / X_HI) * y_scale, y_id),
        ood_error       = ood_relative_error(pred_ood_raw, y_ood),
        ood_error_vs_p1 = e_p1,
        ood_error_vs_p2 = e_p2,
        sc_train        = spectral_complexity(x_train / X_HI, y_train / y_scale),
    )

    # (2) Log-log OLS  — correct for P1
    lx_tr = np.log(x_train)
    A_ll  = np.hstack([lx_tr, np.ones_like(lx_tr)])
    w_ll, _, _, _ = np.linalg.lstsq(A_ll, ly_tr, rcond=None)
    def _pred_ll(x): return np.exp(np.hstack([np.log(x), np.ones_like(x)]) @ w_ll)
    e_p1, e_p2 = _cross_ood(_pred_ll(x_ood), x_ood)
    results["log_log_ols"] = dict(
        id_error        = ood_relative_error(_pred_ll(x_id),  y_id),
        ood_error       = ood_relative_error(_pred_ll(x_ood), y_ood),
        ood_error_vs_p1 = e_p1,
        ood_error_vs_p2 = e_p2,
        sc_train        = spectral_complexity(lx_tr, ly_tr),
    )

    # (3) Log-y OLS  — correct for P2
    A_ly = np.hstack([x_train, np.ones_like(x_train)])
    w_ly, _, _, _ = np.linalg.lstsq(A_ly, ly_tr, rcond=None)
    def _pred_ly(x): return np.exp(np.hstack([x, np.ones_like(x)]) @ w_ly)
    e_p1, e_p2 = _cross_ood(_pred_ly(x_ood), x_ood)
    results["log_y_ols"] = dict(
        id_error        = ood_relative_error(_pred_ly(x_id),  y_id),
        ood_error       = ood_relative_error(_pred_ly(x_ood), y_ood),
        ood_error_vs_p1 = e_p1,
        ood_error_vs_p2 = e_p2,
        sc_train        = spectral_complexity(x_train, ly_tr),
    )

    # (4) Log-log MLP
    ly_scale   = float(np.abs(ly_tr).max())
    lx_tr_norm = lx_tr / np.log(X_HI)
    m_ll = MLP(1)
    train_mlp(m_ll, lx_tr_norm, ly_tr / ly_scale, epochs=epochs)
    def _pred_ll_mlp(x): return np.exp(evaluate_mlp(m_ll, np.log(x) / np.log(X_HI)) * ly_scale)
    e_p1, e_p2 = _cross_ood(_pred_ll_mlp(x_ood), x_ood)
    results["log_log_mlp"] = dict(
        id_error        = ood_relative_error(_pred_ll_mlp(x_id),  y_id),
        ood_error       = ood_relative_error(_pred_ll_mlp(x_ood), y_ood),
        ood_error_vs_p1 = e_p1,
        ood_error_vs_p2 = e_p2,
        sc_train        = spectral_complexity(lx_tr_norm, ly_tr / ly_scale),
    )

    # (5) Log-y MLP
    m_ly = MLP(1)
    train_mlp(m_ly, x_train / X_HI, ly_tr / ly_scale, epochs=epochs)
    def _pred_ly_mlp(x): return np.exp(evaluate_mlp(m_ly, x / X_HI) * ly_scale)
    e_p1, e_p2 = _cross_ood(_pred_ly_mlp(x_ood), x_ood)
    results["log_y_mlp"] = dict(
        id_error        = ood_relative_error(_pred_ly_mlp(x_id),  y_id),
        ood_error       = ood_relative_error(_pred_ly_mlp(x_ood), y_ood),
        ood_error_vs_p1 = e_p1,
        ood_error_vs_p2 = e_p2,
        sc_train        = spectral_complexity(x_train / X_HI, ly_tr / ly_scale),
    )

    # ── SINDy δ_ood diagnostic conditions ───────────────────────────────────
    # Each condition fits a sparse polynomial in a candidate transformed
    # coordinate; δ is the squared-NMSE residual in fit space (matches the toy
    # battery / Kepler / MAK formulation). Three feature maps per DGP: log-log
    # (canonical for power-law), log-y (canonical for exponential), raw.
    import pysindy as ps

    def _sindy_delta(Z_tr, y_tr_fit, Z_id, Z_ood, y_id, y_ood,
                     y_id_fit, y_ood_fit, log_y, degree=2, threshold=0.05):
        lib = ps.PolynomialLibrary(degree=degree, include_bias=True)
        lib.fit(Z_tr)
        Th_tr  = np.asarray(lib.transform(Z_tr))
        Th_id  = np.asarray(lib.transform(Z_id))
        Th_oo  = np.asarray(lib.transform(Z_ood))
        opt = ps.STLSQ(threshold=threshold, alpha=0.0, max_iter=50)
        opt.fit(Th_tr, y_tr_fit.reshape(-1, 1))
        xi = np.asarray(opt.coef_[0])
        pred_tr_fit  = Th_tr @ xi
        pred_id_fit  = Th_id @ xi
        pred_ood_fit = Th_oo @ xi
        dtr = float(np.sum((y_tr_fit  - pred_tr_fit ) ** 2)
                    / (np.sum(y_tr_fit  ** 2) + 1e-12))
        dood = float(np.sum((y_ood_fit - pred_ood_fit) ** 2)
                     / (np.sum(y_ood_fit ** 2) + 1e-12))
        if log_y:
            pred_id  = np.exp(pred_id_fit)
            pred_ood = np.exp(pred_ood_fit)
        else:
            pred_id  = pred_id_fit
            pred_ood = pred_ood_fit
        ie = ood_relative_error(pred_id.flatten(),  y_id)
        oe = ood_relative_error(pred_ood.flatten(), y_ood)
        e_p1, e_p2 = _cross_ood(pred_ood, x_ood)
        n_terms = int((np.abs(xi) > 1e-10).sum())
        return ie, oe, e_p1, e_p2, xi.tolist(), n_terms, dtr, dood

    # Log-log (canonical for power-law DGP P1: y = x^B → log y = B log x).
    # degree=1 — both DGPs are linear in the canonical transform, so the
    # polynomial library only needs to span affine functions; degree=2 admits
    # spurious quadratic fits on noisy data that flip canonical/wrong rankings.
    ie, oe, e_p1, e_p2, xi, k, dtr, dood = _sindy_delta(
        np.log(x_train), ly_tr,
        np.log(x_id), np.log(x_ood), y_id.flatten(), y_ood.flatten(),
        np.log(np.maximum(y_id.flatten(),  1e-8)),
        np.log(np.maximum(y_ood.flatten(), 1e-8)),
        log_y=True, degree=1)
    results["sindy_loglog"] = dict(
        id_error=ie, ood_error=oe,
        ood_error_vs_p1=e_p1, ood_error_vs_p2=e_p2,
        xi=xi, n_terms=k, delta_train=dtr, delta_ood=dood,
    )

    # Log-y (canonical for exponential DGP P2: y = exp(α(x-1)) → log y = α(x-1))
    ie, oe, e_p1, e_p2, xi, k, dtr, dood = _sindy_delta(
        x_train, ly_tr,
        x_id, x_ood, y_id.flatten(), y_ood.flatten(),
        np.log(np.maximum(y_id.flatten(),  1e-8)),
        np.log(np.maximum(y_ood.flatten(), 1e-8)),
        log_y=True, degree=1)
    results["sindy_logy"] = dict(
        id_error=ie, ood_error=oe,
        ood_error_vs_p1=e_p1, ood_error_vs_p2=e_p2,
        xi=xi, n_terms=k, delta_train=dtr, delta_ood=dood,
    )

    # Raw (wrong basis for both DGPs)
    ie, oe, e_p1, e_p2, xi, k, dtr, dood = _sindy_delta(
        x_train, y_train.flatten(),
        x_id, x_ood, y_id.flatten(), y_ood.flatten(),
        y_id.flatten(), y_ood.flatten(),
        log_y=False, degree=1)
    results["sindy_raw"] = dict(
        id_error=ie, ood_error=oe,
        ood_error_vs_p1=e_p1, ood_error_vs_p2=e_p2,
        xi=xi, n_terms=k, delta_train=dtr, delta_ood=dood,
    )

    return results


def run_one_dgp_foundation(dgp_fn, dgp_name, seed,
                           noise_std=0.2, n_train=150, n_test=300,
                           tfm_backend="cpu"):
    """Foundation-model conditions (TabPFN, TimesFM) for one DGP.

    Reuses the same noisy training grid as `run_one_dgp` (so seeds match).
    TabPFN: zero-shot tabular regressor on raw, log-log, log-y features.
    TimesFM: zero-shot forecaster on uniform 128-context grid; OOD horizon
    spans $\\approx [2, 3]$ (partial coverage of regression OOD $[2, 10]$).
    """
    rng = np.random.default_rng(seed)

    x_train = np.linspace(X_LO, X_HI,  n_train).reshape(-1, 1)
    x_id    = np.linspace(X_LO, X_HI,  n_test).reshape(-1, 1)
    x_ood   = np.linspace(X_HI, X_OOD, n_test).reshape(-1, 1)

    y_train_clean = dgp_fn(x_train)
    y_train = y_train_clean + rng.normal(0, noise_std, size=y_train_clean.shape)
    y_id    = dgp_fn(x_id)
    y_ood   = dgp_fn(x_ood)
    ly_tr   = np.log(np.maximum(y_train, 1e-8))

    results = {}

    # ── TabPFN ──────────────────────────────────────────────────────────────
    tpfn = TabPFNModel()
    if tpfn.available:
        # Raw (no commitment)
        tp = TabPFNModel()
        tp.fit(x_train, y_train.flatten())
        pred_id  = tp.predict(x_id)
        pred_ood = tp.predict(x_ood)
        e_p1, e_p2 = _cross_ood(pred_ood, x_ood)
        results["tabpfn_raw"] = dict(
            id_error        = ood_relative_error(pred_id,  y_id),
            ood_error       = ood_relative_error(pred_ood, y_ood),
            ood_error_vs_p1 = e_p1,
            ood_error_vs_p2 = e_p2,
        )

        # Log-log (correct for P1): TabPFN learns log y as a function of log x.
        tp_ll = TabPFNModel()
        tp_ll.fit(np.log(x_train), ly_tr.flatten())
        pred_id_ll  = np.exp(tp_ll.predict(np.log(x_id)))
        pred_ood_ll = np.exp(tp_ll.predict(np.log(x_ood)))
        e_p1, e_p2 = _cross_ood(pred_ood_ll, x_ood)
        results["tabpfn_loglog"] = dict(
            id_error        = ood_relative_error(pred_id_ll,  y_id),
            ood_error       = ood_relative_error(pred_ood_ll, y_ood),
            ood_error_vs_p1 = e_p1,
            ood_error_vs_p2 = e_p2,
        )

        # Log-y (correct for P2): TabPFN learns log y as a function of x.
        tp_ly = TabPFNModel()
        tp_ly.fit(x_train, ly_tr.flatten())
        pred_id_ly  = np.exp(tp_ly.predict(x_id))
        pred_ood_ly = np.exp(tp_ly.predict(x_ood))
        e_p1, e_p2 = _cross_ood(pred_ood_ly, x_ood)
        results["tabpfn_logy"] = dict(
            id_error        = ood_relative_error(pred_id_ly,  y_id),
            ood_error       = ood_relative_error(pred_ood_ly, y_ood),
            ood_error_vs_p1 = e_p1,
            ood_error_vs_p2 = e_p2,
        )
    else:
        print("  TabPFN not available — skipping.")

    # ── TimesFM ─────────────────────────────────────────────────────────────
    # 128-point uniform context on [X_LO, X_HI] = [1, 2] (step ≈ 1/127).
    # Forecast horizon spans the FULL OOD window [X_HI, X_OOD] = [2, 10] at
    # the same step (≈ 1016 points, autoregressed via 8 patches of 128).
    # Three feature-map conditions, mirroring OLS / TabPFN:
    #   • timesfm        : forecast y from y_ctx                         (no feature map)
    #   • timesfm_logy   : forecast log y with x       as dynamic cov    (log-y;   correct for P2)
    #   • timesfm_loglog : forecast log y with log(x)  as dynamic cov    (log-log; correct for P1)
    # The (target = log y, covariate = z) parallel matches OLS exactly:
    # log_y_ols regresses log y on [x, 1]; log_log_ols regresses log y on [log x, 1].
    set_seed(seed)
    x_ctx  = X_LO + STEP_TFM * np.arange(N_CTX_TFM)
    x_fore = X_HI + STEP_TFM * np.arange(1, N_OOD_TFM + 1)
    y_ctx  = dgp_fn(x_ctx)
    y_fore_true = dgp_fn(x_fore)
    ly_ctx = np.log(np.maximum(y_ctx, 1e-8))
    x_all_cov = np.concatenate([x_ctx, x_fore])

    tfm = TimesFMModel(context_len=N_CTX_TFM, horizon_len=HORIZON_TFM,
                       backend=tfm_backend)

    # (a) Raw y forecast (no feature map)
    tfm.set_context(y_ctx.astype(np.float32))
    y_fore_pred = tfm.forecast(N_OOD_TFM)

    results["timesfm"] = dict(
        id_error        = None,
        ood_error       = ood_relative_error(y_fore_pred, y_fore_true),
        ood_error_vs_p1 = ood_relative_error(y_fore_pred, dgp_power(x_fore)),
        ood_error_vs_p2 = ood_relative_error(y_fore_pred, dgp_exp(x_fore)),
        ood_x_range     = [float(x_fore[0]), float(x_fore[-1])],
    )

    # (b) Log-y feature map: target = log y, covariate = x. Correct for exponential P2.
    cov_logy = {"x": x_all_cov.astype(np.float32)}
    ly_fore_pred_logy = tfm.forecast_with_covariates(
        ly_ctx.astype(np.float32), cov_logy, N_OOD_TFM,
    )
    y_fore_pred_logy = np.exp(ly_fore_pred_logy)
    results["timesfm_logy"] = dict(
        id_error        = None,
        ood_error       = ood_relative_error(y_fore_pred_logy, y_fore_true),
        ood_error_vs_p1 = ood_relative_error(y_fore_pred_logy, dgp_power(x_fore)),
        ood_error_vs_p2 = ood_relative_error(y_fore_pred_logy, dgp_exp(x_fore)),
        ood_x_range     = [float(x_fore[0]), float(x_fore[-1])],
    )

    # (c) Log-log feature map: target = log y, covariate = log x. Correct for power-law P1.
    cov_loglog = {"log_x": np.log(x_all_cov).astype(np.float32)}
    ly_fore_pred_loglog = tfm.forecast_with_covariates(
        ly_ctx.astype(np.float32), cov_loglog, N_OOD_TFM,
    )
    y_fore_pred_loglog = np.exp(ly_fore_pred_loglog)
    results["timesfm_loglog"] = dict(
        id_error        = None,
        ood_error       = ood_relative_error(y_fore_pred_loglog, y_fore_true),
        ood_error_vs_p1 = ood_relative_error(y_fore_pred_loglog, dgp_power(x_fore)),
        ood_error_vs_p2 = ood_relative_error(y_fore_pred_loglog, dgp_exp(x_fore)),
        ood_x_range     = [float(x_fore[0]), float(x_fore[-1])],
    )

    return results


def run(seed: int = 42, epochs: int = 2000,
        skip_foundation: bool = False, tfm_backend: str = "cpu",
        save: bool = False):
    set_seed(seed)
    t0 = init_timing()

    x_tr = np.linspace(X_LO, X_HI, 500).reshape(-1, 1)
    diff_train = np.abs(dgp_power(x_tr) - dgp_exp(x_tr))
    x_ood_ = np.linspace(X_HI, X_OOD, 500).reshape(-1, 1)
    diff_ood = np.abs(dgp_power(x_ood_) - dgp_exp(x_ood_))

    print(f"\n=== Exp 1.2  DGP Specificity: Power Law vs. Exponential ===")
    print(f"\n  α = {ALPHA:.4f}  (P1(2)=P2(2)=4)")
    print(f"  Indistinguishability on [{X_LO},{X_HI}]:")
    print(f"    max |P1-P2| = {diff_train.max():.3f}  "
          f"(relative: {diff_train.max()/dgp_power(x_tr).mean()*100:.1f}%)")
    print(f"  OOD divergence [{X_HI},{X_OOD}]: max |P1-P2| = {diff_ood.max():.1f}")

    all_results = {}
    for dgp_name, dgp_fn in [("P1_powerlaw", dgp_power), ("P2_exponential", dgp_exp)]:
        r = run_one_dgp(dgp_fn, dgp_name, seed=seed, epochs=epochs)
        if not skip_foundation:
            print(f"\n  Running TabPFN + TimesFM for {dgp_name} ...")
            r.update(run_one_dgp_foundation(dgp_fn, dgp_name, seed=seed,
                                             tfm_backend=tfm_backend))
        all_results[dgp_name] = r

    print(f"\n  {'DGP':<18}  {'condition':<14}  {'ID err%':>8}  {'OOD err%':>9}  {'SC_train':>9}")
    for dgp_name, r in all_results.items():
        for cond_name, v in r.items():
            ie = v.get("id_error")
            id_s = "--" if ie is None else ("n/a" if (isinstance(ie, float) and np.isnan(ie)) else f"{ie:>8.2f}")
            sc_s = (f"{v['sc_train']:>9.4f}"
                    if "sc_train" in v and not np.isnan(v["sc_train"]) else "      n/a")
            correct = (("log_log" in cond_name and "P1" in dgp_name) or
                       ("log_y"   in cond_name and "P2" in dgp_name))
            marker = "  ← correct" if correct else ""
            print(f"  {dgp_name:<18}  {cond_name:<14}  {id_s:>8}  {v['ood_error']:>9.2f}  {sc_s}{marker}")

    out = dict(
        experiment_id = "1.2",
        name          = "powerlaw_vs_exponential_dgp_specificity",
        seed          = seed,
        epochs        = epochs,
        config        = dict(x_lo=X_LO, x_hi=X_HI, x_ood=X_OOD, b=B, alpha=ALPHA),
        indistinguishability = dict(
            train_max_abs_diff = float(diff_train.max()),
            ood_max_abs_diff   = float(diff_ood.max()),
        ),
        dgps = all_results,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--skip-foundation", action="store_true",
                        help="Skip TabPFN + TimesFM (regression / SINDy only).")
    parser.add_argument("--tfm-backend", type=str, default="cpu",
                        choices=["cpu", "gpu"])
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs,
        skip_foundation=args.skip_foundation,
        tfm_backend=args.tfm_backend, save=args.save)
