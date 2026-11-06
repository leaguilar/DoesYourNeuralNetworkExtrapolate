"""
Exp RW1 — Real-world validation: Kepler's third law (NASA exoplanet archive)
============================================================================
Kepler's third law:  T = a^(3/2) / sqrt(M_star / M_sun)

Dataset: juliensimon/nasa-exoplanets (Hugging Face), 6,158 confirmed exoplanets.
Stellar subset: rows with (a, T, M_star, R_star, T_eff) all valid = 2,367.

Train / OOD split:  a < 0.5 AU (short-period, easier to detect) / 0.5–20 AU.
Real scatter (~14% in log-log space) comes from stellar mass heterogeneity.

Ablation design: [Variable subset] × [Log transform] × [Model class]

Conditions (all 4 models: OLS, SINDy, MLP, TabPFN)
--------------------------------------------------
{a}:
  raw_*      : (a) → T
  log_a_*    : (log a) → log T

{a, M*}:
  raw_aM_*      : (a, M*) → T              [bivariate raw]
  log_a_linM_*  : (log a, M*) → log T      [wrong M* transform]
  lina_logM_*   : (a, log M*) → log T      [wrong a transform]
  log_aM_*      : (log a, log M*) → log T  [CORRECT — Kepler]

{a, R*}:
  log_aR_*   : (log a, log R*) → log T     [wrong mass proxy]

{a, T_eff}:
  log_aTe_*  : (log a, log T_eff) → log T  [worse proxy]

{a, M*, R*, T_eff}:
  log_aMRTe_* : (log a, log M*, log R*, log T_eff) → log T
"""

import argparse
import math
import os
import sys

import numpy as np
import pysindy as ps

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_seed, save_result, ood_relative_error, MLP, train_mlp, evaluate_mlp,
                   init_timing, collect_meta)

# ---------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------

A_SPLIT      = 0.5    # AU; training window  a < A_SPLIT
A_OOD_MAX    = 20.0   # AU; cap OOD region
DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_kepler_data():
    """Return dicts (train, ood) with arrays: a, T, Ms, Rs, Te.

    Subset: rows where a, T, M_star, R_star, T_eff are all valid.
    """
    from datasets import load_dataset
    ds = load_dataset("juliensimon/nasa-exoplanets", split="train")
    import pandas as pd
    df = ds.to_pandas()

    cols = ["pl_orbsmax", "pl_orbper", "st_mass", "st_rad", "st_teff"]
    sub  = df[cols].copy()
    sub["pl_orbper"] = sub["pl_orbper"] / DAYS_PER_YEAR

    # Require core stellar variables to be positive
    core_mask = (
        sub["pl_orbsmax"].notna() & (sub["pl_orbsmax"] > 0) &
        sub["pl_orbper"].notna()  & (sub["pl_orbper"]  > 0) &
        sub["st_mass"].notna()    & (sub["st_mass"]    > 0) &
        sub["st_rad"].notna()     & (sub["st_rad"]     > 0) &
        sub["st_teff"].notna()    & (sub["st_teff"]    > 0)
    )
    sub = sub[core_mask].reset_index(drop=True)

    a  = sub["pl_orbsmax"].values
    T  = sub["pl_orbper"].values
    Ms = sub["st_mass"].values
    Rs = sub["st_rad"].values
    Te = sub["st_teff"].values

    mask_tr  = a < A_SPLIT
    mask_ood = (a >= A_SPLIT) & (a < A_OOD_MAX)

    def _split(arr, m1, m2):
        return arr[m1], arr[m2]

    tr  = {k: v for k, v in zip(["a","T","Ms","Rs","Te"],
                                  [_split(x, mask_tr, mask_ood)[0]
                                   for x in [a, T, Ms, Rs, Te]])}
    ood = {k: v for k, v in zip(["a","T","Ms","Rs","Te"],
                                  [_split(x, mask_tr, mask_ood)[1]
                                   for x in [a, T, Ms, Rs, Te]])}
    return tr, ood


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------

def _ols(X_tr, y_tr, X_id, X_ood, y_id, y_ood, log_y=False):
    """Fit OLS, return (id_error, ood_error, coef) in original y space."""
    Xa = np.column_stack([X_tr, np.ones(len(X_tr))])
    Xi = np.column_stack([X_id,  np.ones(len(X_id))])
    Xo = np.column_stack([X_ood, np.ones(len(X_ood))])
    w, _, _, _ = np.linalg.lstsq(Xa, y_tr, rcond=None)
    pred_id  = Xi @ w
    pred_ood = Xo @ w
    if log_y:
        pred_id  = np.exp(pred_id)
        pred_ood = np.exp(pred_ood)
    ie = ood_relative_error(pred_id,  y_id)
    oe = ood_relative_error(pred_ood, y_ood)
    return ie, oe, w[:-1].tolist()   # drop intercept from coef


# ---------------------------------------------------------------------------
# SINDy helpers (static polynomial regression) — uses pysindy
# ---------------------------------------------------------------------------
#
# Ad-hoc implementations kept below as comments for reference.
#
# from itertools import combinations_with_replacement
#
# def _build_library(Z, degree):
#     N, d = Z.shape
#     seen = set(); cols = [np.ones(N)]
#     for deg in range(1, degree + 1):
#         for combo in combinations_with_replacement(range(d), deg):
#             key = tuple(sorted(combo))
#             if key in seen: continue
#             seen.add(key)
#             cols.append(np.prod([Z[:, k] for k in combo], axis=0))
#     return np.column_stack(cols)
#
# def _stlsq(Theta, y, threshold=0.05, max_iter=50):
#     xi = np.linalg.lstsq(Theta, y, rcond=None)[0]
#     for _ in range(max_iter):
#         mask = np.abs(xi) >= threshold
#         if not mask.any(): return np.zeros_like(xi)
#         xi_new = np.zeros_like(xi)
#         xi_new[mask] = np.linalg.lstsq(Theta[:, mask], y, rcond=None)[0]
#         if np.allclose(xi, xi_new, atol=1e-10): break
#         xi = xi_new
#     return xi


def _sindy_static(Z_tr, y_tr, Z_id, Z_ood, y_id, y_ood,
                  degree=2, threshold=0.05, log_y=False):
    """Sparse polynomial regression y = Theta(Z) @ xi via pysindy.

    Returns (id_error, ood_error, xi, n_terms, delta_train, delta_ood):
      - id/ood_error: relative error in raw y space (after exp if log_y).
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

    pred_tr_fit  = Th_tr @ xi
    pred_id_fit  = Th_id @ xi
    pred_ood_fit = Th_oo @ xi

    y_ood_fit = np.log(y_ood) if log_y else y_ood
    delta_train = float(np.sum((y_tr      - pred_tr_fit ) ** 2)
                        / (np.sum(y_tr      ** 2) + 1e-12))
    delta_ood   = float(np.sum((y_ood_fit - pred_ood_fit) ** 2)
                        / (np.sum(y_ood_fit ** 2) + 1e-12))

    if log_y:
        pred_id  = np.exp(pred_id_fit)
        pred_ood = np.exp(pred_ood_fit)
    else:
        pred_id  = pred_id_fit
        pred_ood = pred_ood_fit
    ie = ood_relative_error(pred_id,  y_id)
    oe = ood_relative_error(pred_ood, y_ood)
    n_terms = int((np.abs(xi) > 1e-10).sum())
    return ie, oe, xi.tolist(), n_terms, delta_train, delta_ood


# ---------------------------------------------------------------------------
# TabPFN helper
# ---------------------------------------------------------------------------

_TABPFN_TOKEN = None
_here  = os.path.dirname(__file__)
_env_p = os.path.join(_here, "..", "..", ".env")
_env_t = os.environ.get("TABPFN_TOKEN")
if _env_t:
    _TABPFN_TOKEN = _env_t
elif os.path.isfile(_env_p):
    with open(_env_p) as _f:
        for _line in _f:
            if _line.strip().startswith("TABPFN_TOKEN"):
                _TABPFN_TOKEN = _line.split("=", 1)[1].strip().strip('"').strip("'")
                break

_TabPFN = None
if _TABPFN_TOKEN:
    try:
        import tabpfn_client
        tabpfn_client.set_access_token(_TABPFN_TOKEN)
        from tabpfn_client import TabPFNRegressor as _TabPFNRegressor
        _TabPFN = _TabPFNRegressor
    except Exception:
        pass


def _tabpfn(X_tr, y_tr, X_id, X_ood, y_id, y_ood, log_y=False):
    nan = float("nan")
    if _TabPFN is None:
        return nan, nan
    m = _TabPFN(ignore_pretraining_limits=True)
    m.fit(X_tr, y_tr.flatten())
    pred_id  = m.predict(X_id).flatten()
    pred_ood = m.predict(X_ood).flatten()
    if log_y:
        pred_id  = np.exp(pred_id)
        pred_ood = np.exp(pred_ood)
    return (ood_relative_error(pred_id,  y_id),
            ood_relative_error(pred_ood, y_ood))


# ---------------------------------------------------------------------------
# Main condition runner
# ---------------------------------------------------------------------------

def run_conditions(tr, ood, seed, epochs):
    a_tr,  T_tr,  Ms_tr,  Rs_tr,  Te_tr = (
        tr["a"], tr["T"], tr["Ms"], tr["Rs"], tr["Te"])
    a_oo,  T_oo,  Ms_oo,  Rs_oo,  Te_oo = (
        ood["a"], ood["T"], ood["Ms"], ood["Rs"], ood["Te"])

    la_tr  = np.log(a_tr);  lMs_tr = np.log(Ms_tr)
    lRs_tr = np.log(Rs_tr); lTe_tr = np.log(Te_tr); lT_tr = np.log(T_tr)
    la_oo  = np.log(a_oo);  lMs_oo = np.log(Ms_oo)
    lRs_oo = np.log(Rs_oo); lTe_oo = np.log(Te_oo); lT_oo = np.log(T_oo)

    res = {}

    # ------------------------------------------------------------------ OLS --
    # 1. raw: (a) → T
    ie, oe, w = _ols(a_tr.reshape(-1,1), T_tr,
                     a_tr.reshape(-1,1), a_oo.reshape(-1,1), T_tr, T_oo)
    res["raw_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # 2. log(a) → log T  [correct univariate]
    ie, oe, w = _ols(la_tr.reshape(-1,1), lT_tr,
                     la_tr.reshape(-1,1), la_oo.reshape(-1,1), T_tr, T_oo, log_y=True)
    res["log_a_ols"] = dict(id_error=ie, ood_error=oe, coef=w,
                            slope=round(w[0], 4))

    # Bivariate raw (a, M*) feature matrices — reused across model classes
    X_tr_rawM = np.column_stack([a_tr,  Ms_tr])
    X_oo_rawM = np.column_stack([a_oo,  Ms_oo])

    # 3. (a, M*) → T  [bivariate, no log]
    ie, oe, w = _ols(X_tr_rawM, T_tr, X_tr_rawM, X_oo_rawM, T_tr, T_oo)
    res["raw_aM_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # 4. (log a, log M*) → log T  [CORRECT — Kepler]
    X_tr2 = np.column_stack([la_tr, lMs_tr])
    X_oo2 = np.column_stack([la_oo, lMs_oo])
    ie, oe, w = _ols(X_tr2, lT_tr, X_tr2, X_oo2, T_tr, T_oo, log_y=True)
    res["log_aM_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # 4. (log a, M*) → log T  [wrong M* transform]
    X_tr_linM = np.column_stack([la_tr, Ms_tr])
    X_oo_linM = np.column_stack([la_oo, Ms_oo])
    ie, oe, w = _ols(X_tr_linM, lT_tr, X_tr_linM, X_oo_linM, T_tr, T_oo, log_y=True)
    res["log_a_linM_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # 5. (a, log M*) → log T  [wrong a transform — diverges OOD]
    X_tr_linaM = np.column_stack([a_tr, lMs_tr])
    X_oo_linaM = np.column_stack([a_oo, lMs_oo])
    ie, oe, w = _ols(X_tr_linaM, lT_tr, X_tr_linaM, X_oo_linaM, T_tr, T_oo, log_y=True)
    res["lina_logM_ols"] = dict(
        id_error=ie,
        ood_error=min(oe, 1e6),   # cap runaway divergence for table display
        ood_error_raw=oe,
        coef=w,
    )

    # 6. (log a, log R*) → log T  [wrong mass proxy]
    X_tr_R = np.column_stack([la_tr, lRs_tr])
    X_oo_R = np.column_stack([la_oo, lRs_oo])
    ie, oe, w = _ols(X_tr_R, lT_tr, X_tr_R, X_oo_R, T_tr, T_oo, log_y=True)
    res["log_aR_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # 7. (log a, log T_eff) → log T  [worse mass proxy]
    X_tr_Te = np.column_stack([la_tr, lTe_tr])
    X_oo_Te = np.column_stack([la_oo, lTe_oo])
    ie, oe, w = _ols(X_tr_Te, lT_tr, X_tr_Te, X_oo_Te, T_tr, T_oo, log_y=True)
    res["log_aTe_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # 8. (log a, log M*, log R*, log T_eff) → log T  [all stellar]
    X_tr_all = np.column_stack([la_tr, lMs_tr, lRs_tr, lTe_tr])
    X_oo_all = np.column_stack([la_oo, lMs_oo, lRs_oo, lTe_oo])
    ie, oe, w = _ols(X_tr_all, lT_tr, X_tr_all, X_oo_all, T_tr, T_oo, log_y=True)
    res["log_aMRTe_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # --------------------------------------------------------------- SINDy --
    # degree=2 polynomial library; threshold tuned so that irrelevant monomials
    # are zeroed out while the true Kepler terms survive.

    # 10. sparse poly of (log a) → log T
    Z_tr1 = la_tr.reshape(-1, 1); Z_oo1 = la_oo.reshape(-1, 1)
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Z_tr1, lT_tr, Z_tr1, Z_oo1, T_tr, T_oo,
        degree=2, threshold=0.05, log_y=True)
    res["log_a_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                               delta_train=dtr, delta_ood=dood)

    # 11. sparse poly of (log a, log M*) → log T  [should recover (1.5, -0.5)]
    Z_tr2 = np.column_stack([la_tr, lMs_tr])
    Z_oo2 = np.column_stack([la_oo, lMs_oo])
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Z_tr2, lT_tr, Z_tr2, Z_oo2, T_tr, T_oo,
        degree=2, threshold=0.05, log_y=True)
    res["log_aM_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                                delta_train=dtr, delta_ood=dood)

    # raw_aM_sindy: sparse poly of (a, M*) → T  [bivariate, no log]
    ie, oe, xi, k, dtr, dood = _sindy_static(
        X_tr_rawM, T_tr, X_tr_rawM, X_oo_rawM, T_tr, T_oo,
        degree=2, threshold=0.05)
    res["raw_aM_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                                delta_train=dtr, delta_ood=dood)

    # --------------------------------------------------------------- MLP ---
    x_sc = float(a_tr.max())
    y_sc = float(np.abs(T_tr).max())

    # 12. MLP on raw (a, M*, R*, T_eff)
    Xr_tr = np.column_stack([a_tr, Ms_tr, Rs_tr, Te_tr])
    Xr_oo = np.column_stack([a_oo, Ms_oo, Rs_oo, Te_oo])
    Xs_tr = Xr_tr / (Xr_tr.max(axis=0) + 1e-8)
    Xs_oo = Xr_oo / (Xr_tr.max(axis=0) + 1e-8)
    m = MLP(4)
    train_mlp(m, Xs_tr, T_tr / y_sc, epochs=epochs)
    res["raw_mlp"] = dict(
        id_error  = ood_relative_error(evaluate_mlp(m, Xs_tr) * y_sc, T_tr),
        ood_error = ood_relative_error(evaluate_mlp(m, Xs_oo) * y_sc, T_oo),
    )

    # raw_aM_mlp: MLP on raw (a, M*) → T  [bivariate, no log]
    aM_col_max = X_tr_rawM.max(axis=0) + 1e-8
    Xs_aM_tr = X_tr_rawM / aM_col_max
    Xs_aM_oo = X_oo_rawM / aM_col_max
    m_rawAM = MLP(2)
    train_mlp(m_rawAM, Xs_aM_tr, T_tr / y_sc, epochs=epochs)
    res["raw_aM_mlp"] = dict(
        id_error  = ood_relative_error(evaluate_mlp(m_rawAM, Xs_aM_tr) * y_sc, T_tr),
        ood_error = ood_relative_error(evaluate_mlp(m_rawAM, Xs_aM_oo) * y_sc, T_oo),
    )

    # 13. MLP on (log a, log M*)
    lx_min = float(la_tr.min()); lx_range = float(la_tr.max() - la_tr.min()) + 1e-8
    lM_min = float(lMs_tr.min()); lM_range = float(lMs_tr.max() - lMs_tr.min()) + 1e-8
    ly_min = float(lT_tr.min()); ly_range = float(lT_tr.max() - lT_tr.min()) + 1e-8

    def _norm2(la_, lM_):
        return np.column_stack([(la_ - lx_min) / lx_range,
                                 (lM_ - lM_min) / lM_range])

    m2 = MLP(2)
    train_mlp(m2, _norm2(la_tr, lMs_tr), (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp2(la_, lM_):
        return np.exp(evaluate_mlp(m2, _norm2(la_, lM_)) * ly_range + ly_min)

    res["log_aM_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp2(la_tr, lMs_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp2(la_oo, lMs_oo), T_oo),
    )

    # 14. MLP on (log a)  [univariate correct]
    def _norm1d(la_):
        return ((la_ - lx_min) / lx_range).reshape(-1, 1)

    m_la = MLP(1)
    train_mlp(m_la, _norm1d(la_tr), (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp_la(la_):
        return np.exp(evaluate_mlp(m_la, _norm1d(la_)) * ly_range + ly_min)

    res["log_a_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_la(la_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp_la(la_oo), T_oo),
    )

    # 15. MLP on (log a, M*)  [wrong M* transform]
    a_min_tr = float(a_tr.min()); a_rng_tr = float(a_tr.max() - a_tr.min()) + 1e-8
    Ms_min_tr = float(Ms_tr.min()); Ms_rng_tr = float(Ms_tr.max() - Ms_tr.min()) + 1e-8

    def _norm_lalinM(la_, Ms_):
        return np.column_stack([(la_ - lx_min) / lx_range,
                                 (Ms_ - Ms_min_tr) / Ms_rng_tr])

    m_lalinM = MLP(2)
    train_mlp(m_lalinM, _norm_lalinM(la_tr, Ms_tr), (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp_lalinM(la_, Ms_):
        return np.exp(evaluate_mlp(m_lalinM, _norm_lalinM(la_, Ms_)) * ly_range + ly_min)

    res["log_a_linM_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_lalinM(la_tr, Ms_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp_lalinM(la_oo, Ms_oo), T_oo),
    )

    # 16. MLP on (a, log M*)  [wrong a transform]
    def _norm_linaM(a_, lM_):
        return np.column_stack([(a_ - a_min_tr) / a_rng_tr,
                                 (lM_ - lM_min) / lM_range])

    m_linaM = MLP(2)
    train_mlp(m_linaM, _norm_linaM(a_tr, lMs_tr), (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp_linaM(a_, lM_):
        raw = evaluate_mlp(m_linaM, _norm_linaM(a_, lM_)) * ly_range + ly_min
        return np.exp(raw)

    oe_linaM = ood_relative_error(_pred_mlp_linaM(a_oo, lMs_oo), T_oo)
    res["lina_logM_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_linaM(a_tr, lMs_tr), T_tr),
        ood_error = min(oe_linaM, 1e6),
        ood_error_raw = oe_linaM,
    )

    # 17. MLP on all stellar (log)  [(log a, log M*, log R*, log T_eff)]
    all_min = X_tr_all.min(axis=0); all_rng = X_tr_all.max(axis=0) - all_min + 1e-8
    Xs_all_tr = (X_tr_all - all_min) / all_rng
    Xs_all_oo = (X_oo_all - all_min) / all_rng
    m_all = MLP(4)
    train_mlp(m_all, Xs_all_tr, (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp_all(Xs_):
        return np.exp(evaluate_mlp(m_all, Xs_) * ly_range + ly_min)

    res["log_aMRTe_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_all(Xs_all_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp_all(Xs_all_oo), T_oo),
    )

    # 18. OLS on raw all stellar (a, M*, R*, T_eff) → T
    ie, oe, w = _ols(Xr_tr, T_tr, Xr_tr, Xr_oo, T_tr, T_oo)
    res["raw_all_ols"] = dict(id_error=ie, ood_error=oe, coef=w)

    # -------------------------------------------------------------- TabPFN --
    # 19. TabPFN raw (a, M*, R*, T_eff) → T
    ie_t, oe_t = _tabpfn(Xs_tr, T_tr, Xs_tr, Xs_oo, T_tr, T_oo)
    res["raw_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # 20. TabPFN (log a) → log T  [univariate correct]
    ie_t, oe_t = _tabpfn(la_tr.reshape(-1,1), lT_tr,
                          la_tr.reshape(-1,1), la_oo.reshape(-1,1),
                          T_tr, T_oo, log_y=True)
    res["log_a_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # 21. TabPFN (log a, log M*) → log T
    ie_t, oe_t = _tabpfn(np.column_stack([la_tr, lMs_tr]), lT_tr,
                          np.column_stack([la_tr, lMs_tr]),
                          np.column_stack([la_oo, lMs_oo]),
                          T_tr, T_oo, log_y=True)
    res["log_aM_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # 22. TabPFN (log a, M*) → log T  [wrong M* transform]
    ie_t, oe_t = _tabpfn(np.column_stack([la_tr, Ms_tr]), lT_tr,
                          np.column_stack([la_tr, Ms_tr]),
                          np.column_stack([la_oo, Ms_oo]),
                          T_tr, T_oo, log_y=True)
    res["log_a_linM_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # 23. TabPFN (a, log M*) → log T  [wrong a transform]
    ie_t, oe_t = _tabpfn(np.column_stack([a_tr, lMs_tr]), lT_tr,
                          np.column_stack([a_tr, lMs_tr]),
                          np.column_stack([a_oo, lMs_oo]),
                          T_tr, T_oo, log_y=True)
    oe_linaM_t = oe_t if not math.isnan(oe_t) else oe_t
    res["lina_logM_tabpfn"] = dict(id_error=ie_t,
                                    ood_error=min(oe_linaM_t, 1e6) if math.isfinite(oe_linaM_t) else oe_linaM_t)

    # 24. TabPFN (log a, log M*, log R*, log T_eff) → log T
    ie_t, oe_t = _tabpfn(X_tr_all, lT_tr, X_tr_all, X_oo_all, T_tr, T_oo, log_y=True)
    res["log_aMRTe_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # ─── Missing SINDy conditions ────────────────────────────────────────────
    # 25. SINDy on raw (a) → T  [no log]
    ie, oe, xi, k, dtr, dood = _sindy_static(
        a_tr.reshape(-1,1), T_tr, a_tr.reshape(-1,1), a_oo.reshape(-1,1), T_tr, T_oo,
        degree=2, threshold=0.05)
    res["raw_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                             delta_train=dtr, delta_ood=dood)

    # 26. SINDy (log a, M*) → log T  [wrong M* transform — polynomial M* terms diverge]
    Z_lalinM_tr = np.column_stack([la_tr, Ms_tr])
    Z_lalinM_oo = np.column_stack([la_oo, Ms_oo])
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Z_lalinM_tr, lT_tr, Z_lalinM_tr, Z_lalinM_oo,
        T_tr, T_oo, degree=2, threshold=0.05, log_y=True)
    oe_cap = min(oe, 1e6) if math.isfinite(oe) else oe
    res["log_a_linM_sindy"] = dict(id_error=ie, ood_error=oe_cap, ood_error_raw=oe,
                                    xi=xi, n_terms=k,
                                    delta_train=dtr, delta_ood=dood)

    # 27. SINDy (a, log M*) → log T  [wrong a transform]
    Z_linaM_tr = np.column_stack([a_tr, lMs_tr])
    Z_linaM_oo = np.column_stack([a_oo, lMs_oo])
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Z_linaM_tr, lT_tr, Z_linaM_tr, Z_linaM_oo,
        T_tr, T_oo, degree=2, threshold=0.05, log_y=True)
    oe_cap = min(oe, 1e6)
    res["lina_logM_sindy"] = dict(id_error=ie, ood_error=oe_cap, ood_error_raw=oe,
                                   xi=xi, n_terms=k,
                                   delta_train=dtr, delta_ood=dood)

    # 28. SINDy (log a, log R*) → log T  [proxy R*]
    Z_aR_tr = np.column_stack([la_tr, lRs_tr])
    Z_aR_oo = np.column_stack([la_oo, lRs_oo])
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Z_aR_tr, lT_tr, Z_aR_tr, Z_aR_oo,
        T_tr, T_oo, degree=2, threshold=0.05, log_y=True)
    res["log_aR_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                                delta_train=dtr, delta_ood=dood)

    # 29. SINDy (log a, log T_eff) → log T  [proxy T_eff]
    Z_aTe_tr = np.column_stack([la_tr, lTe_tr])
    Z_aTe_oo = np.column_stack([la_oo, lTe_oo])
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Z_aTe_tr, lT_tr, Z_aTe_tr, Z_aTe_oo,
        T_tr, T_oo, degree=2, threshold=0.05, log_y=True)
    res["log_aTe_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                                 delta_train=dtr, delta_ood=dood)

    # 30. SINDy all stellar (log) → log T
    ie, oe, xi, k, dtr, dood = _sindy_static(
        X_tr_all, lT_tr, X_tr_all, X_oo_all,
        T_tr, T_oo, degree=2, threshold=0.05, log_y=True)
    res["log_aMRTe_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                                   delta_train=dtr, delta_ood=dood)

    # 32. SINDy raw all stellar (a, M*, R*, T_eff) → T
    ie, oe, xi, k, dtr, dood = _sindy_static(
        Xr_tr, T_tr, Xr_tr, Xr_oo,
        T_tr, T_oo, degree=2, threshold=0.05)
    res["raw_all_sindy"] = dict(id_error=ie, ood_error=oe, xi=xi, n_terms=k,
                                 delta_train=dtr, delta_ood=dood)

    # ─── Missing MLP conditions ──────────────────────────────────────────────
    # 33. MLP on raw (a) → T  [univariate]
    a_min_tr_ = float(a_tr.min()); a_rng_tr_ = float(a_tr.max() - a_tr.min()) + 1e-8
    y_sc_ = float(T_tr.max()) + 1e-8

    m_ra = MLP(1)
    train_mlp(m_ra, ((a_tr - a_min_tr_) / a_rng_tr_).reshape(-1,1),
              T_tr / y_sc_, epochs=epochs)

    def _pred_mlp_ra(a_):
        return evaluate_mlp(m_ra, ((a_ - a_min_tr_) / a_rng_tr_).reshape(-1,1)) * y_sc_

    res["raw_a_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_ra(a_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp_ra(a_oo), T_oo),
    )

    # 34. MLP (log a, log R*) → log T  [proxy R*]
    lRs_min = float(lRs_tr.min()); lRs_rng = float(lRs_tr.max() - lRs_tr.min()) + 1e-8

    def _norm_aR(la_, lR_):
        return np.column_stack([(la_ - lx_min) / lx_range, (lR_ - lRs_min) / lRs_rng])

    m_aR = MLP(2)
    train_mlp(m_aR, _norm_aR(la_tr, lRs_tr), (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp_aR(la_, lR_):
        return np.exp(evaluate_mlp(m_aR, _norm_aR(la_, lR_)) * ly_range + ly_min)

    res["log_aR_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_aR(la_tr, lRs_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp_aR(la_oo, lRs_oo), T_oo),
    )

    # 35. MLP (log a, log T_eff) → log T  [proxy T_eff]
    lTe_min = float(lTe_tr.min()); lTe_rng = float(lTe_tr.max() - lTe_tr.min()) + 1e-8

    def _norm_aTe(la_, lTe_):
        return np.column_stack([(la_ - lx_min) / lx_range, (lTe_ - lTe_min) / lTe_rng])

    m_aTe = MLP(2)
    train_mlp(m_aTe, _norm_aTe(la_tr, lTe_tr), (lT_tr - ly_min) / ly_range, epochs=epochs)

    def _pred_mlp_aTe(la_, lTe_):
        return np.exp(evaluate_mlp(m_aTe, _norm_aTe(la_, lTe_)) * ly_range + ly_min)

    res["log_aTe_mlp"] = dict(
        id_error  = ood_relative_error(_pred_mlp_aTe(la_tr, lTe_tr), T_tr),
        ood_error = ood_relative_error(_pred_mlp_aTe(la_oo, lTe_oo), T_oo),
    )

    # ─── Missing TabPFN conditions ───────────────────────────────────────────
    # 37. TabPFN raw (a) → T  [univariate]
    ie_t, oe_t = _tabpfn(a_tr.reshape(-1,1), T_tr,
                          a_tr.reshape(-1,1), a_oo.reshape(-1,1), T_tr, T_oo)
    res["raw_a_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # raw_aM_tabpfn: TabPFN on raw (a, M*) → T  [bivariate, no log]
    ie_t, oe_t = _tabpfn(X_tr_rawM, T_tr, X_tr_rawM, X_oo_rawM, T_tr, T_oo)
    res["raw_aM_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # 38. TabPFN (log a, log R*) → log T  [proxy R*]
    ie_t, oe_t = _tabpfn(np.column_stack([la_tr, lRs_tr]), lT_tr,
                          np.column_stack([la_tr, lRs_tr]),
                          np.column_stack([la_oo, lRs_oo]),
                          T_tr, T_oo, log_y=True)
    res["log_aR_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    # 39. TabPFN (log a, log T_eff) → log T  [proxy T_eff]
    ie_t, oe_t = _tabpfn(np.column_stack([la_tr, lTe_tr]), lT_tr,
                          np.column_stack([la_tr, lTe_tr]),
                          np.column_stack([la_oo, lTe_oo]),
                          T_tr, T_oo, log_y=True)
    res["log_aTe_tabpfn"] = dict(id_error=ie_t, ood_error=oe_t)

    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seed: int = 42, epochs: int = 3000, save: bool = False):
    set_seed(seed)
    t0 = init_timing()
    print(f"\n=== Exp RW1  Kepler's Third Law — multivariate (seed={seed}) ===")
    print("  Loading dataset ...")
    tr, ood = load_kepler_data()
    print(f"  Stellar subset: train n={len(tr['a'])}, OOD n={len(ood['a'])}")
    print(f"  a train: [{tr['a'].min():.4f}, {tr['a'].max():.4f}] AU")
    print(f"  a OOD:   [{ood['a'].min():.3f}, {ood['a'].max():.2f}] AU")

    la = np.log(tr["a"]); lM = np.log(tr["Ms"]); lT = np.log(tr["T"])
    Ac = np.column_stack([la, lM, np.ones_like(la)])
    w_chk, _, _, _ = np.linalg.lstsq(Ac, lT, rcond=None)
    print(f"\n  OLS log T ~ b_a*log a + b_M*log M  (train only):")
    print(f"    b_a={w_chk[0]:.3f}  b_M={w_chk[1]:.3f}  (Kepler: 1.500, -0.500)")

    results = run_conditions(tr, ood, seed=seed, epochs=epochs)

    print(f"\n  {'condition':<22}  {'ID err%':>8}  {'OOD err%':>9}")
    for cond, v in results.items():
        ie = v.get("id_error", math.nan)
        oe = v.get("ood_error", math.nan)
        ie_s = f"{ie:>8.1f}" if math.isfinite(ie) else "     nan"
        oe_s = f"{oe:>9.1f}" if math.isfinite(oe) and oe < 1e5 else "  >1e5" if math.isfinite(oe) else "      nan"
        print(f"  {cond:<22}  {ie_s}  {oe_s}")

    out = dict(
        experiment_id = "2.2",
        name          = "kepler_third_law_multivariate",
        seed          = seed,
        epochs        = epochs,
        config        = dict(
            a_split   = A_SPLIT,
            a_ood_max = A_OOD_MAX,
            n_train   = int(len(tr["a"])),
            n_ood     = int(len(ood["a"])),
        ),
        results = results,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, epochs=args.epochs, save=args.save)
