"""
Exp 1.3 (revised) — Parsimonious governing equations as coordinate diagnostic
==============================================================================
Replaces the SC-gap diagnostic.

Motivation (Champion et al., PNAS 2019):
  "This approach fundamentally relies on an effective coordinate system in
  which the dynamics have a simple representation."

Core claim:
  The correct feature map is the one in which the governing equation
  dv/d(indep) = Theta(z) @ xi
  can be expressed with the FEWEST active terms AND generalises to OOD data.
  A wrong feature map either requires many terms or the fitted xi fails OOD.

Metric — SINDy-OOD score:
  For each (DGP, feature map) pair:
    1. Map: z = phi(indep_var), v = psi(y)
    2. Compute dv/d(indep) numerically on training data
    3. Build polynomial library Theta(z) (degree chosen per feature family)
    4. Fit sparse xi via STLSQ on training data
    5. Evaluate Theta(z_ood) @ xi vs true dv/d(indep)_ood
    6. Report: k (active terms), train_deriv_res, ood_deriv_res

  ood_deriv_res = 0.0 iff the governing equation is globally exact in this
  coordinate system and generalises OOD.  Wrong maps give ood_deriv_res >> 0.

DGPs tested:
  Fourier family   sin(x), sin(2x)
  Poly family      x^2, x^3+x
  G1               sin(x)cos(2x)    [Fourier manifold, Exp 2.5]
  G2               tanh(5sin(x)cos(2x))
  I1               log y = 2u-u^2, u=log(x)   [log-polynomial, Exp 1.3]
  I2               log y = 1-2e^{-x}
"""

import argparse
import math
import os
import sys

import numpy as np
import pysindy as ps
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from utils import set_seed, save_result, init_timing, collect_meta

def sindy_ood_score(indep_tr, v_tr, z_tr,
                    indep_oo, v_oo, z_oo,
                    degree: int = 2, threshold: float = 0.05) -> dict:
    """
    Fit dv/d(indep) = Theta(z) @ xi on training data; evaluate on OOD.

    Uses pysindy.PolynomialLibrary for Theta and pysindy.STLSQ for xi.

    Parameters
    ----------
    indep_tr / indep_oo : 1-D arrays — independent variable (x or log x)
    v_tr / v_oo         : 1-D arrays — target state (y or log y)
    z_tr / z_oo         : (N, d) arrays — feature representation
    degree              : polynomial degree for library
    threshold           : STLSQ sparsity threshold

    Returns
    -------
    dict with keys: n_terms, train_deriv_res, ood_deriv_res, xi (list)
    """
    if z_tr.ndim == 1:
        z_tr = z_tr[:, None]
    if z_oo.ndim == 1:
        z_oo = z_oo[:, None]

    dv_tr = np.gradient(v_tr, indep_tr)
    dv_oo = np.gradient(v_oo, indep_oo)

    lib = ps.PolynomialLibrary(degree=degree, include_bias=True)
    lib.fit(z_tr)
    Th_tr = np.asarray(lib.transform(z_tr))
    Th_oo = np.asarray(lib.transform(z_oo))

    opt = ps.STLSQ(threshold=threshold, alpha=0.0, max_iter=50)
    opt.fit(Th_tr, dv_tr.reshape(-1, 1))
    xi = np.asarray(opt.coef_[0])

    n_terms = int((np.abs(xi) > 1e-10).sum())
    pred_tr = Th_tr @ xi
    pred_oo = Th_oo @ xi

    tr_res = float(np.sum((dv_tr - pred_tr) ** 2) /
                   (np.sum(dv_tr ** 2) + 1e-12))
    oo_res = float(np.sum((dv_oo - pred_oo) ** 2) /
                   (np.sum(dv_oo ** 2) + 1e-12))

    active = {int(i): float(xi[i]) for i in range(len(xi)) if abs(xi[i]) > 1e-10}
    return dict(n_terms=n_terms, train_deriv_res=tr_res,
                ood_deriv_res=oo_res, xi=active)


# ---------------------------------------------------------------------------
# DGPs and feature maps
# ---------------------------------------------------------------------------

def _fourier4(x):
    return np.c_[np.sin(x), np.cos(x), np.sin(2 * x), np.cos(2 * x)]

def _fourier6(x):
    return np.c_[np.sin(x), np.cos(x),
                 np.sin(2 * x), np.cos(2 * x),
                 np.sin(3 * x), np.cos(3 * x)]


# Each entry:
#   key        — unique string id
#   dgp_name   — display label
#   indep_fn   — callable: x_raw -> independent variable
#   state_fn   — callable: (x_raw, y_raw) -> state v
#   feature_fn — callable: (x_raw, y_raw) -> feature matrix z
#   degree     — library degree (1 for poly/log, 2 for Fourier)
#   x_tr_lo/hi, x_oo_hi  — raw x ranges
#   correct    — True iff this (DGP, transform) pair has a sparse exact model

CASES = []

def add_case(dgp, tf, dgp_fn, indep_fn, state_fn, feat_fn, degree,
             x_tr_lo, x_tr_hi, x_oo_hi, correct):
    CASES.append(dict(dgp=dgp, tf=tf, dgp_fn=dgp_fn,
                      indep_fn=indep_fn, state_fn=state_fn,
                      feat_fn=feat_fn, degree=degree,
                      x_tr_lo=x_tr_lo, x_tr_hi=x_tr_hi,
                      x_oo_hi=x_oo_hi, correct=correct))


# ── Fourier family: sin(x), sin(2x) ──────────────────────────────────────────
for _dgp, _fn, _corr in [
    ("sin(x)",  np.sin,                "fourier"),
    ("sin(2x)", lambda x: np.sin(2*x), "fourier2"),
]:
    for _tf, _ifn, _sfn, _ffn, _deg in [
        ("raw",      lambda x: x,  lambda x,y: y, lambda x,y: x[:,None],                           1),
        ("fourier",  lambda x: x,  lambda x,y: y, lambda x,y: np.c_[np.sin(x), np.cos(x)],         2),
        ("fourier2", lambda x: x,  lambda x,y: y, lambda x,y: np.c_[np.sin(2*x), np.cos(2*x)],     2),
        ("poly2",    lambda x: x,  lambda x,y: y, lambda x,y: np.c_[x, x**2],                      1),
        ("poly3",    lambda x: x,  lambda x,y: y, lambda x,y: np.c_[x, x**2, x**3],                1),
    ]:
        add_case(_dgp, _tf, _fn, _ifn, _sfn, _ffn, _deg,
                 0.1, 0.9, 5.0, _tf == _corr)

# ── Poly family: x², x³+x ────────────────────────────────────────────────────
# Correctness = the derivative IS polynomial in x at the required degree.
# x²:   d/dx(x²) = 2x  → degree-1 → raw, poly2, poly3 all work
# x³+x: d/dx(x³+x) = 3x²+1 → degree-2 → poly2 and poly3 both work
_poly_correct = {
    "x^2":   {"raw", "poly2", "poly3"},
    "x^3+x": {"poly2", "poly3"},
}
for _dgp, _fn in [
    ("x^2",   lambda x: x**2),
    ("x^3+x", lambda x: x**3 + x),
]:
    for _tf, _ifn, _sfn, _ffn, _deg in [
        ("raw",      lambda x: x, lambda x,y: y, lambda x,y: x[:,None],            1),
        ("fourier",  lambda x: x, lambda x,y: y, lambda x,y: np.c_[np.sin(x), np.cos(x)],   2),
        ("poly2",    lambda x: x, lambda x,y: y, lambda x,y: np.c_[x, x**2],       1),
        ("poly3",    lambda x: x, lambda x,y: y, lambda x,y: np.c_[x, x**2, x**3], 1),
    ]:
        add_case(_dgp, _tf, _fn, _ifn, _sfn, _ffn, _deg,
                 0.1, 0.9, 5.0, _tf in _poly_correct[_dgp])

# ── G1: sin(x)cos(2x) ────────────────────────────────────────────────────────
_G1 = lambda x: np.sin(x) * np.cos(2 * x)
for _tf, _ffn, _deg, _ok in [
    ("raw",       lambda x,y: x[:,None],  1, False),
    # fourier-4 with degree-2 lib: cos(3x)=cos(x)cos(2x)-sin(x)sin(2x) in span → correct
    ("fourier-4", lambda x,y: _fourier4(x), 2, True),
    ("fourier-6", lambda x,y: _fourier6(x), 2, True),
    ("poly3",     lambda x,y: np.c_[x, x**2, x**3], 1, False),
]:
    add_case("G1:sin(x)cos(2x)", _tf, _G1,
             lambda x: x, lambda x,y: y, _ffn, _deg,
             -math.pi, math.pi, 4*math.pi, _ok)

# ── G2: tanh(5·sin(x)·cos(2x)) ───────────────────────────────────────────────
_G2 = lambda x: np.tanh(5 * np.sin(x) * np.cos(2 * x))
for _tf, _ffn, _deg in [
    ("raw",       lambda x,y: x[:,None],    1),
    ("fourier-4", lambda x,y: _fourier4(x), 2),
    ("fourier-6", lambda x,y: _fourier6(x), 2),
    ("poly3",     lambda x,y: np.c_[x, x**2, x**3], 1),
]:
    add_case("G2:tanh(5sincos)", _tf, _G2,
             lambda x: x, lambda x,y: y, _ffn, _deg,
             -math.pi, math.pi, 4*math.pi, False)  # no sparse OLS repr exists

# ── I1: log y = 2u - u², u = log(x) ─────────────────────────────────────────
# Independent variable: u = log(x); state: v = log(y)
# Governing eq: dv/du = 2 - 2u  (2-term sparse in [1, u])
_I1 = lambda x: np.exp(2 * np.log(x) - np.log(x) ** 2)
for _tf, _ifn, _sfn, _ffn, _deg, _ok in [
    # wrong: raw x, state y
    ("raw x->y",      lambda x: x,         lambda x,y: y,         lambda x,y: x[:,None],                      1, False),
    # wrong: log-x as indep, state y (not log y)
    ("logx,raw y",    lambda x: np.log(x), lambda x,y: y,         lambda x,y: np.log(x)[:,None],              2, False),
    # correct: u=log(x) indep, v=log(y) state, z=[u] with deg-2 lib covers [1,u,u²]
    ("loglog",        lambda x: np.log(x), lambda x,y: np.log(y), lambda x,y: np.log(x)[:,None],              2, True),
    # wrong: polynomial in raw x, state y
    ("poly2 x->y",    lambda x: x,         lambda x,y: y,         lambda x,y: np.c_[x, x**2],                 1, False),
]:
    add_case("I1:logy=2u-u²", _tf, _I1,
             _ifn, _sfn, _ffn, _deg,
             0.5, 2.0, 8.0, _ok)

# ── I2: log y = 1 - 2e^{-x} ──────────────────────────────────────────────────
# Independent variable: x; state: v = log(y)
# Governing eq: dv/dx = 1 - v  (2-term sparse in [1, v] — autonomous ODE)
_I2 = lambda x: np.exp(1 - 2 * np.exp(-x))
for _tf, _ifn, _sfn, _ffn, _deg, _ok in [
    # wrong: raw x, state y
    ("raw x->y",      lambda x: x, lambda x,y: y,         lambda x,y: x[:,None],                    1, False),
    # wrong: x as feature, v=log(y) as state — dv/dx=2e^{-x} not sparse in poly(x)
    ("logy,x-feat",   lambda x: x, lambda x,y: np.log(y), lambda x,y: x[:,None],                    1, False),
    # correct: v=log(y) as feature (autonomous: dv/dx = 1-v)
    ("logy,v-state",  lambda x: x, lambda x,y: np.log(y), lambda x,y: np.log(y)[:,None],            1, True),
    # also correct: e^{-x} as feature: dv/dx = 2z, z=e^{-x} → sparse in z
    ("logy,e^-x",     lambda x: x, lambda x,y: np.log(y), lambda x,y: np.exp(-x)[:,None],           1, True),
    # wrong: poly2 in x, state y
    ("poly2 x->y",    lambda x: x, lambda x,y: y,         lambda x,y: np.c_[x, x**2],               1, False),
]:
    add_case("I2:logy=1-2e^{-x}", _tf, _I2,
             _ifn, _sfn, _ffn, _deg,
             0.0, 2.0, 8.0, _ok)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

N_TR = 200
N_OO = 400


def run(seed: int = 42, save: bool = False) -> dict:
    set_seed(seed)
    t0 = init_timing()

    records = {}
    all_ood_d  = []
    all_correct = []

    print(f"\n=== Exp 1.3 SINDy Coordinate Diagnostic (seed={seed}) ===")
    print(f"\n  {'DGP':<24}  {'Transform':<22}  {'k':>3}  {'tr_d':>7}  {'ood_d':>7}  {'correct':>7}")
    print("  " + "-" * 80)

    current_dgp = None
    for case in CASES:
        dgp = case["dgp"]
        tf  = case["tf"]

        # Build data
        x_oo_lo = case["x_tr_hi"]  # OOD starts where training ends
        x_tr = np.linspace(case["x_tr_lo"], case["x_tr_hi"], N_TR)
        x_oo = np.linspace(x_oo_lo,         case["x_oo_hi"],  N_OO)
        y_tr = case["dgp_fn"](x_tr)
        y_oo = case["dgp_fn"](x_oo)

        indep_tr = case["indep_fn"](x_tr)
        indep_oo = case["indep_fn"](x_oo)
        v_tr = case["state_fn"](x_tr, y_tr)
        v_oo = case["state_fn"](x_oo, y_oo)
        z_tr = case["feat_fn"](x_tr, y_tr)
        z_oo = case["feat_fn"](x_oo, y_oo)

        res = sindy_ood_score(indep_tr, v_tr, z_tr,
                              indep_oo, v_oo, z_oo,
                              degree=case["degree"])

        if dgp not in records:
            records[dgp] = {}
        records[dgp][tf] = dict(
            correct=case["correct"],
            n_terms=res["n_terms"],
            train_deriv_res=res["train_deriv_res"],
            ood_deriv_res=res["ood_deriv_res"],
            xi=res["xi"],
        )

        all_ood_d.append(res["ood_deriv_res"])
        all_correct.append(case["correct"])

        if dgp != current_dgp:
            print()
            current_dgp = dgp

        marker = "✓" if case["correct"] else ""
        ood_s = f"{res['ood_deriv_res']:>7.4f}" if math.isfinite(res["ood_deriv_res"]) else "    inf"
        print(f"  {dgp:<24}  {tf:<22}  {res['n_terms']:>3}  "
              f"{res['train_deriv_res']:>7.4f}  {ood_s}  {marker:>7}")

    # Spearman: ood_deriv_res vs binary correctness label
    # (1=wrong, 0=correct — ood_d should be 0 for correct, high for wrong)
    all_ood_d   = np.array(all_ood_d)
    all_correct = np.array(all_correct)
    wrong_label = (~all_correct).astype(float)
    finite = np.isfinite(all_ood_d)
    rho, pval = spearmanr(all_ood_d[finite], wrong_label[finite])

    print(f"\n  Total pairs: {len(CASES)}")
    print(f"  Spearman ρ(ood_deriv_res, wrong_label) = {rho:+.3f}  (p={pval:.3e})")
    print(f"  ood_deriv_res for CORRECT transforms: "
          f"mean={all_ood_d[all_correct].mean():.4f}  "
          f"max={all_ood_d[all_correct].max():.4f}")
    print(f"  ood_deriv_res for WRONG transforms:   "
          f"mean={all_ood_d[~all_correct].mean():.2f}  "
          f"min={all_ood_d[~all_correct].min():.4f}")

    out = dict(
        experiment_id="1.3",
        name="sindy_coordinate_diagnostic",
        seed=seed,
        spearman=dict(
            rho=float(rho), pval=float(pval),
            n_pairs=int(finite.sum()),
        ),
        records=records,
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    run(seed=args.seed, save=args.save)
