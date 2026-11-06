#!/usr/bin/env python3
"""
Exp 3.1 — near-boundary segment WIDTH sweep and the t_V collapse
================================================================
Companion to `exp3_1_selection_protocol.py`. That experiment fixes the
near-boundary window at [0, π] (as wide as the training window) and asks
whether selection beats cross-validation. This one asks *how thin the
segment may get*, i.e. it sweeps the segment width w down towards the
purely geometric floor ε/r of Prop. 2 (\\Cref{prop:nearboundary}) and past
it, and then checks that the resulting accuracy surface is governed by the
single signal-to-noise statistic t_V of Prop. 3 (\\Cref{prop:strip}):

    t_V := sqrt(n_V) * ||Δ||_{L2(D_V)} / σ,     Δ := P_1 - P_2 ,

with Δ measured, per cell, between the oracle-best candidate's fit and its
*hardest wrong competitor* on the segment, so that the four-way selection
problem is scored through the two-point subproblem Prop. 3 controls.

Two passes, deliberately kept separate (they consume different random
streams and answer different questions):

  WIDTH pass     σ ∈ {0.01, 0.1}; both clean-label and noisy-label segments;
                 reports selection accuracy per (σ, label regime, w, n_V)
                 cell against the 5-fold-CV baseline and against the Prop. 2
                 geometric floor ε/r.
  COLLAPSE pass  σ ∈ {0.003, 0.01, 0.03, 0.1}; noisy-label segments only;
                 records t_V per cell so that the 144 = 9 × 4 × 4 cells can
                 be binned by t_V alone.

The training fit does not depend on (w, n_V), so each trial fits the library
once per σ and evaluates every (w, n_V) cell against that fit.

Accuracy is always measured against the *oracle* selection (the candidate
with the lowest OOD error on [π, 3π]), matching Exp. 3.1's convention.

Usage
-----
    conda run -n data_analysis python src/experiments/exp3_1_width_sweep.py \\
        --seed 42 --n_trials 100 --save
    # ... repeat for seeds 43, 44, then pool:
    conda run -n data_analysis python src/experiments/exp3_1_width_sweep.py --report
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (set_seed, save_result, ood_relative_error,   # noqa: E402
                   init_timing, collect_meta)
from exp3_1_selection_protocol import (                          # noqa: E402
    LIBRARY, CORRECT, K, RANDOM_BASE,
    f_true, make_features, ols_fit, ols_predict, cv_score,
    X_TRAIN_LO, X_TRAIN_HI, X_BNDRY_HI, X_OOD_HI,
)

# ---- Sweep grid (frozen; the paper numbers are for exactly this grid) ------
WIDTHS          = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, float(np.pi))
NVALS           = (10, 25, 50, 100)
SIGMAS_WIDTH    = (0.01, 0.1)
SIGMAS_COLLAPSE = (0.003, 0.01, 0.03, 0.1)
REGIMES         = ("clean", "noisy")
N_TRAIN         = 200
N_TRIALS        = 100
SEEDS_PAPER     = (42, 43, 44)

# Practical threshold for the relative-error selection rule (Prop. 3(b) gives
# the LR-test threshold t_V = 2Φ^{-1}(acc); the four-way relative-error rule
# used here crosses the CV baseline later, at t_V ≈ 6).
T_RULE    = 6.0
BIN_EDGES = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, float("inf"))

EID  = "3.1_width_sweep"
NAME = "near_boundary_width_sweep_and_tV_collapse"


def _ck(*parts) -> str:
    """Canonical cell key, e.g. (0.01, 'noisy', 0.16, 100) -> '0.01|noisy|0.16|100'."""
    return "|".join(("%g" % p) if isinstance(p, float) else str(p) for p in parts)


# ---- Prop. 2 geometric floor ----------------------------------------------

def eps_over_r(n_grid: int = 4000) -> dict:
    """Geometric floor per polynomial candidate, from the fitted in-window gap.

    The paper is stated throughout in L2, so the floor is too:

      eps_W = ||sin - poly_deg||_{L2(D_train)}   (in-window agreement level)
      r     = initial divergence rate of |Delta| just outside the window
      floor = sqrt(3) * eps_W / r  -- the width at which the strip's L2
              separation first matches the window's, i.e. t_V >= t_W sqrt(n_V/n).

    The fitting range for r is t in (0, 0.02], and the floor is sensitive to it.
    Deterministic — no RNG involved.
    """
    out = {}
    W = np.linspace(X_TRAIN_LO, X_TRAIN_HI, n_grid)
    f = f_true(W)
    for deg, nm in ((7, "poly7_ols"), (9, "poly9_ols")):
        c = np.polyfit(W, f, deg)
        resid = f - np.polyval(c, W)
        eps_w = float(np.sqrt(np.mean(resid ** 2)))
        t = np.linspace(0, 0.5, 5000)
        gap = np.abs(f_true(t) - np.polyval(c, t))
        r = float(np.polyfit(t[1:200], gap[1:200], 1)[0])
        out[nm] = dict(eps_w=eps_w, r=r,
                       floor_l2=float(np.sqrt(3.0) * eps_w / r))
    out["floor"] = max(v["floor_l2"] for v in out.values() if isinstance(v, dict))
    return out


# ---- Pass 1: width × segment size, clean and noisy segment labels ----------

def run_width_pass(seed: int, n_trials: int, n_train: int) -> dict:
    """σ ∈ SIGMAS_WIDTH, both label regimes. Returns per-cell correct counts."""
    hits = {}
    cv_hits = {s: 0 for s in SIGMAS_WIDTH}
    for trial in range(n_trials):
        rng = np.random.default_rng(seed * 10000 + trial)
        x_tr = np.linspace(X_TRAIN_LO, X_TRAIN_HI, n_train).reshape(-1, 1)
        x_oo = np.linspace(X_BNDRY_HI, X_OOD_HI, n_train).reshape(-1, 1)
        y_oo = f_true(x_oo)
        for sigma in SIGMAS_WIDTH:
            y_tr = f_true(x_tr) + rng.normal(0, sigma, n_train)
            fits, oracle = {}, {}
            for nm in LIBRARY:
                w_ = ols_fit(make_features(nm, x_tr), y_tr)
                fits[nm] = w_
                oracle[nm] = ood_relative_error(
                    ols_predict(make_features(nm, x_oo), w_), y_oo)
            orc = min(oracle, key=oracle.get)
            cvs = {nm: cv_score(nm, x_tr, y_tr) for nm in LIBRARY}
            cv_hits[sigma] += int(min(cvs, key=cvs.get) == orc)
            for w in WIDTHS:
                for nv in NVALS:
                    x_v = np.linspace(X_TRAIN_HI, X_TRAIN_HI + w, nv).reshape(-1, 1)
                    y_clean = f_true(x_v)
                    for regime in REGIMES:
                        y_v = (y_clean if regime == "clean"
                               else y_clean + rng.normal(0, sigma, nv))
                        sc = {nm: ood_relative_error(
                            ols_predict(make_features(nm, x_v), fits[nm]), y_v)
                            for nm in LIBRARY}
                        k = _ck(sigma, regime, w, nv)
                        hits[k] = hits.get(k, 0) + int(min(sc, key=sc.get) == orc)
    return dict(cv_correct={_ck(s): cv_hits[s] for s in SIGMAS_WIDTH},
                cells={k: dict(correct=v) for k, v in hits.items()})


# ---- Pass 2: the t_V collapse ---------------------------------------------

def run_collapse_pass(seed: int, n_trials: int, n_train: int) -> dict:
    """σ ∈ SIGMAS_COLLAPSE, noisy segment labels, t_V recorded per cell."""
    hits, tsum = {}, {}
    cv_hits = {s: 0 for s in SIGMAS_COLLAPSE}
    for trial in range(n_trials):
        rng = np.random.default_rng(seed * 10000 + trial)
        x_tr = np.linspace(X_TRAIN_LO, X_TRAIN_HI, n_train).reshape(-1, 1)
        x_oo = np.linspace(X_BNDRY_HI, X_OOD_HI, n_train).reshape(-1, 1)
        y_oo = f_true(x_oo)
        for sigma in SIGMAS_COLLAPSE:
            y_tr = f_true(x_tr) + rng.normal(0, sigma, n_train)
            fits = {nm: ols_fit(make_features(nm, x_tr), y_tr) for nm in LIBRARY}
            orc = min(LIBRARY, key=lambda nm: ood_relative_error(
                ols_predict(make_features(nm, x_oo), fits[nm]), y_oo))
            cvs = {nm: cv_score(nm, x_tr, y_tr) for nm in LIBRARY}
            cv_hits[sigma] += int(min(cvs, key=cvs.get) == orc)
            for w in WIDTHS:
                for nv in NVALS:
                    x_v = np.linspace(X_TRAIN_HI, X_TRAIN_HI + w, nv).reshape(-1, 1)
                    P = {nm: ols_predict(make_features(nm, x_v), fits[nm])
                         for nm in LIBRARY}
                    # ||Δ||_{L2(D_V)} against the hardest wrong competitor
                    gap = min(float(np.sqrt(np.mean((P[orc] - P[nm]) ** 2)))
                              for nm in LIBRARY if nm != orc)
                    t_v = gap * np.sqrt(nv) / sigma
                    y_v = f_true(x_v) + rng.normal(0, sigma, nv)
                    sc = {nm: ood_relative_error(P[nm], y_v) for nm in LIBRARY}
                    k = _ck(sigma, w, nv)
                    hits[k] = hits.get(k, 0) + int(min(sc, key=sc.get) == orc)
                    tsum[k] = tsum.get(k, 0.0) + float(t_v)
    return dict(cv_correct={_ck(s): cv_hits[s] for s in SIGMAS_COLLAPSE},
                cells={k: dict(correct=hits[k], t_sum=tsum[k]) for k in hits})


# ---- Pooling across seeds --------------------------------------------------

def summarize(runs: list) -> dict:
    """Pool per-seed runs into the quantities quoted in the paper.

    `runs` is a list of result dicts as written by `run(save=True)`.
    """
    n = sum(r["n_trials"] for r in runs)
    geo = runs[0]["geometry"]

    def _pool(block, key):
        tot = {}
        for r in runs:
            for k, v in r["results"][block][key].items():
                if isinstance(v, dict):
                    d = tot.setdefault(k, dict(correct=0, t_sum=0.0))
                    d["correct"] += v["correct"]
                    d["t_sum"] += v.get("t_sum", 0.0)
                else:
                    tot[k] = tot.get(k, 0) + v
        return tot

    width_cv = {k: v / n for k, v in _pool("width", "cv_correct").items()}
    width_acc = {k: v["correct"] / n for k, v in _pool("width", "cells").items()}
    coll_cv = {k: v / n for k, v in _pool("collapse", "cv_correct").items()}
    coll_raw = _pool("collapse", "cells")
    coll = {k: dict(acc=v["correct"] / n, t=v["t_sum"] / n) for k, v in coll_raw.items()}

    # CV reference used by the paper: the width pass at σ = 0.01.
    cv_ref = width_cv[_ck(SIGMAS_WIDTH[0])]

    cells = sorted(((v["t"], v["acc"], k) for k, v in coll.items()))
    bins = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        sel = [c for c in cells if lo <= c[0] < hi]
        if not sel:
            continue
        a = [c[1] for c in sel]
        parts = [c[2].split("|") for c in sel]
        bins.append(dict(lo=lo, hi=hi, n=len(sel),
                         acc=float(np.mean(a)), sd=float(np.std(a)),
                         n_sigma=len({p[0] for p in parts}),
                         n_w=len({p[1] for p in parts}),
                         n_nv=len({p[2] for p in parts})))

    above = [c for c in cells if c[0] >= T_RULE]
    below = [c for c in cells if c[0] < T_RULE]
    own = {}
    for t, a, k in cells:
        sig = k.split("|")[0]
        own[k] = coll_cv[sig]

    best = (0, None)
    for thr in np.unique([c[0] for c in cells]):
        corr = sum((c[0] >= thr) == (c[1] > cv_ref) for c in cells)
        if corr > best[0]:
            best = (corr, float(thr))

    slice16 = sorted([c for c in cells if c[2].split("|")[1] == "0.16"])
    ts = [c[0] for c in slice16]
    accs = [c[1] for c in slice16]

    return dict(
        n_trials_pooled=n, n_seeds=len(runs), seeds=[r["seed"] for r in runs],
        geometry=geo, cv_ref=cv_ref,
        width_cv=width_cv, width_acc=width_acc,
        collapse_cv=coll_cv, collapse=coll,
        n_cells=len(cells), bins=bins,
        max_bin_sd=max(b["sd"] for b in bins),
        rule=dict(threshold=T_RULE,
                  n_above=len(above),
                  n_above_beat=sum(c[1] > cv_ref for c in above),
                  n_below=len(below),
                  n_below_nobeat=sum(c[1] <= cv_ref for c in below),
                  n_above_beat_own=sum(c[1] > own[c[2]] for c in above),
                  n_below_nobeat_own=sum(c[1] <= own[c[2]] for c in below)),
        best_threshold=dict(value=best[1], correct=best[0], total=len(cells)),
        slice_w016=dict(t_min=min(ts), t_max=max(ts),
                        acc_min=min(accs), acc_max=max(accs),
                        corr=float(np.corrcoef(ts, accs)[0, 1]),
                        rows=[dict(t=t, acc=a, key=k) for t, a, k in slice16]),
    )


# ---- Reporting -------------------------------------------------------------

def print_report(s: dict):
    geo = s["geometry"]
    n = s["n_trials_pooled"]
    print("\n=== Exp 3.1 width sweep — pooled over seeds %s (%d trials/cell) ==="
          % (s["seeds"], n))
    print("\nGeometric floor (L2):")
    for nm in ("poly7_ols", "poly9_ols"):
        v = geo[nm]
        print("  %-10s eps_W=%.3e  r=%.3e  floor=%.4f"
              % (nm, v["eps_w"], v["r"], v["floor_l2"]))
    print("  binding floor sqrt(3) eps_W / r = %.4f" % geo["floor"])

    for sigma in SIGMAS_WIDTH:
        print("\n=== sigma=%.3g   CV baseline=%.3f ===" % (sigma, s["width_cv"][_ck(sigma)]))
        for regime in REGIMES:
            print("  segment labels: %s" % regime.upper())
            print("    %-8s" % "w" + "".join("%9s" % ("n_V=%d" % v) for v in NVALS)
                  + "   vs floor")
            for w in WIDTHS:
                row = "".join("%9.3f" % s["width_acc"][_ck(sigma, regime, w, v)]
                              for v in NVALS)
                mark = "  <- below sqrt(3) eps_W / r" if w < geo["floor"] else ""
                print("    %-8.3g" % w + row + mark)

    print("\nCV baseline per sigma (collapse pass):")
    for sigma in SIGMAS_COLLAPSE:
        print("  sigma=%-6g CV=%.3f" % (sigma, s["collapse_cv"][_ck(sigma)]))

    print("\ntotal cells: %d" % s["n_cells"])
    print("\nAccuracy vs t_V = sqrt(n_V)*||Delta||_L2(D_V)/sigma")
    print("(each bin pools cells with DIFFERENT w, n_V and sigma)\n")
    print("%-12s %6s %9s %8s %s" % ("t_V range", "cells", "acc mean", "acc sd",
                                    "distinct w / n_V / sigma"))
    for b in s["bins"]:
        rng_s = "<%g" % b["hi"] if b["lo"] == 0 else (
            ">=%g" % b["lo"] if np.isinf(b["hi"]) else "%g-%g" % (b["lo"], b["hi"]))
        print("%-12s %6d %9.3f %8.3f  %d / %d / %d"
              % (rng_s, b["n"], b["acc"], b["sd"], b["n_w"], b["n_nv"], b["n_sigma"]))
    print("max within-bin sd = %.3f" % s["max_bin_sd"])

    r = s["rule"]
    print("\nThreshold t_V >= %g, 'beats CV' = acc > %.3f (fixed):" % (r["threshold"], s["cv_ref"]))
    print("  above: %d cells, %d beat" % (r["n_above"], r["n_above_beat"]))
    print("  below: %d cells, %d do NOT beat" % (r["n_below"], r["n_below_nobeat"]))
    print("  same split vs each cell's own-sigma CV: %d / %d"
          % (r["n_above_beat_own"], r["n_below_nobeat_own"]))
    bt = s["best_threshold"]
    print("Best single threshold: %.2f -> %d/%d correct"
          % (bt["value"], bt["correct"], bt["total"]))

    sl = s["slice_w016"]
    print("\nFixed w=0.16 slice (all sigma, n_V):")
    print("  t_V range %.1f -> %.1f, acc range %.2f -> %.2f, corr=%.2f"
          % (sl["t_min"], sl["t_max"], sl["acc_min"], sl["acc_max"], sl["corr"]))
    for row in sl["rows"]:
        sig, w, nv = row["key"].split("|")
        print("    sig=%-6s n_V=%-4s t=%6.1f acc=%.3f" % (sig, nv, row["t"], row["acc"]))

    print("\nw=0.04 cells (just above the geometric floor):")
    for t, a, k in sorted([(v["t"], v["acc"], k) for k, v in s["collapse"].items()
                           if k.split("|")[1] == "0.04"],
                          key=lambda c: (float(c[2].split("|")[0]),
                                         int(c[2].split("|")[2]))):
        sig, w, nv = k.split("|")
        print("    sig=%-6s n_V=%-4s t=%6.1f acc=%.3f" % (sig, nv, t, a))


def load_runs(results_dir: str = None) -> list:
    """Load every saved seed file of this experiment."""
    import glob
    import json
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "results", "exp" + EID.replace(".", "_"))
    keep = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "exp*.json"))):
        parts = os.path.basename(p)[:-5].split("__")
        if len(parts) < 3:
            continue
        keep[(parts[0], parts[1])] = p           # latest timestamp per seed wins
    runs = []
    for p in keep.values():
        with open(p) as f:
            runs.append(json.load(f))
    return sorted(runs, key=lambda r: r["seed"])


# ---- Entry point -----------------------------------------------------------

def run(seed: int = 42, n_trials: int = N_TRIALS, n_train: int = N_TRAIN,
        save: bool = False) -> dict:
    set_seed(seed)
    t0 = init_timing()

    geo = eps_over_r()
    print(f"\n=== Exp 3.1  Near-boundary width sweep and t_V collapse ===")
    print(f"  DGP sin(x), library {LIBRARY} (K={K}, random baseline {RANDOM_BASE:.2f})")
    print(f"  Training window [{X_TRAIN_LO:.3f}, {X_TRAIN_HI:.3f}] (width "
          f"{X_TRAIN_HI - X_TRAIN_LO:.4f}), n_train={n_train}")
    print(f"  Segment V = [0, w],  w in {[round(w, 3) for w in WIDTHS]}")
    print(f"  n_V in {list(NVALS)}")
    print(f"  width pass sigma {list(SIGMAS_WIDTH)}, "
          f"collapse pass sigma {list(SIGMAS_COLLAPSE)}")
    print(f"  seed={seed}  n_trials={n_trials}")
    print(f"  geometric floor sqrt(3) eps_W / r = {geo['floor']:.4f} "
          f"(poly7 {geo['poly7_ols']['floor_l2']:.4f}, "
          f"poly9 {geo['poly9_ols']['floor_l2']:.4f})")

    print("  running width pass ...")
    width = run_width_pass(seed, n_trials, n_train)
    print("  running collapse pass ...")
    collapse = run_collapse_pass(seed, n_trials, n_train)

    out = dict(
        experiment_id=EID, name=NAME, seed=seed, n_trials=n_trials,
        config=dict(
            widths=list(WIDTHS), n_vals=list(NVALS),
            sigmas_width=list(SIGMAS_WIDTH), sigmas_collapse=list(SIGMAS_COLLAPSE),
            regimes=list(REGIMES), n_train=n_train, library=list(LIBRARY),
            correct=CORRECT, K=K, random_baseline=RANDOM_BASE,
            x_train=[X_TRAIN_LO, X_TRAIN_HI], x_ood=[X_BNDRY_HI, X_OOD_HI],
            train_window_width=float(X_TRAIN_HI - X_TRAIN_LO),
            t_rule=T_RULE, bin_edges=list(BIN_EDGES),
        ),
        geometry=geo,
        results=dict(width=width, collapse=collapse),
    )
    out.update(collect_meta(t0))
    if save:
        save_result(out)

    # Single-seed preview; the paper numbers pool seeds 42/43/44 via --report.
    print_report(summarize([out]))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS)
    parser.add_argument("--n_train",  type=int, default=N_TRAIN)
    parser.add_argument("--save",     action="store_true")
    parser.add_argument("--report",   action="store_true",
                        help="pool every saved seed file and print the paper numbers")
    parser.add_argument("--results_dir", default=None)
    args = parser.parse_args()

    if args.report:
        runs = load_runs(args.results_dir)
        if not runs:
            sys.exit("no saved runs found — run with --save first")
        print_report(summarize(runs))
    else:
        run(seed=args.seed, n_trials=args.n_trials, n_train=args.n_train,
            save=args.save)
