#!/usr/bin/env python3
"""Exp 3.1 (near-boundary width sweep): accuracy vs (w, n_V) and the t_V collapse.

Two blocks in one table:
  (i)  selection accuracy as a function of the segment width w and the segment
       size n_V at a representative noise level, with the Prop. 2 geometric
       floor sqrt(3) eps_W / r marked and the 5-fold CV baseline as the reference;
  (ii) the same 144 (w, n_V, sigma) cells binned by the Prop. 3 statistic
       t_V = sqrt(n_V) ||Delta||_{L2(D_V)} / sigma alone -- each bin pools
       cells built from different widths, segment sizes and noise levels.

Input: results/exp3_1_width_sweep/ (written by
       src/experiments/exp3_1_width_sweep.py, one file per seed).

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from pathlib import Path

import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _loaders import load_results
from _fmt import table_footer, table_header

EID = "3.1_width_sweep"
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp3_1_width_sweep.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"

# Representative cell shown in block (i): the honest comparison to CV is the
# segment that carries the same observation noise as the training data.
SIGMA_REF = 0.01
REGIME_REF = "noisy"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


def _ck(*parts):
    """Cell key, matching exp3_1_width_sweep._ck."""
    return "|".join(("%g" % p) if isinstance(p, float) else str(p) for p in parts)


def _pool(runs):
    """Pool per-seed hit counts into accuracies and mean t_V per cell."""
    n = sum(r["n_trials"] for r in runs)
    width_hits, width_cv, coll = {}, {}, {}
    for r in runs:
        res = r["results"]
        for k, v in res["width"]["cells"].items():
            width_hits[k] = width_hits.get(k, 0) + v["correct"]
        for k, v in res["width"]["cv_correct"].items():
            width_cv[k] = width_cv.get(k, 0) + v
        for k, v in res["collapse"]["cells"].items():
            d = coll.setdefault(k, [0, 0.0])
            d[0] += v["correct"]
            d[1] += v["t_sum"]
    return dict(
        n=n,
        width_acc={k: v / n for k, v in width_hits.items()},
        width_cv={k: v / n for k, v in width_cv.items()},
        collapse={k: dict(acc=v[0] / n, t=v[1] / n) for k, v in coll.items()},
    )


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print("  [table_exp3_1_width_sweep] no input found, skipping")
        return None

    runs = sorted(runs, key=lambda r: r["seed"])
    cfg = runs[0]["config"]
    geo = runs[0]["geometry"]
    P = _pool(runs)

    widths = cfg["widths"]
    nvals = cfg["n_vals"]
    edges = cfg["bin_edges"]
    t_rule = cfg["t_rule"]
    win = cfg["train_window_width"]
    floor = geo["floor"]
    cv_ref = P["width_cv"][_ck(SIGMA_REF)]

    # ---- block (ii): bin every (sigma, w, n_V) cell by its mean t_V ---------
    cells = sorted((v["t"], v["acc"], k) for k, v in P["collapse"].items())
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [c for c in cells if lo <= c[0] < hi]
        if not sel:
            continue
        parts = [c[2].split("|") for c in sel]
        bins.append(dict(
            lo=lo, hi=hi, n=len(sel),
            acc=float(np.mean([c[1] for c in sel])),
            sd=float(np.std([c[1] for c in sel])),
            n_w=len({p[1] for p in parts}),
            n_nv=len({p[2] for p in parts}),
            n_sig=len({p[0] for p in parts}),
        ))
    above = [c for c in cells if c[0] >= t_rule]
    below = [c for c in cells if c[0] < t_rule]
    n_beat = sum(c[1] > cv_ref for c in above)
    n_nobeat = sum(c[1] <= cv_ref for c in below)
    max_sd = max(b["sd"] for b in bins)

    seeds = ", ".join(str(r["seed"]) for r in runs)
    n_sig_coll = len(cfg["sigmas_collapse"])
    sig_list = ", ".join(("%g" % s) for s in cfg["sigmas_collapse"])

    cap = (
        r"Exp.~3.1 (near-boundary width sweep). "
        r"Block~(i): selection accuracy on a near-boundary segment "
        r"$V=[0, w]$ carrying $n_V$ points, "
        r"at $\sigma=10^{-2}$ with segment labels carrying the same noise "
        r"as the training data. "
        rf"Training window $W=[-\pi, 0]$, $|W|=\pi$, $n_{{\mathrm{{train}}}}={cfg['n_train']}$; "
        r"accuracy is agreement with the oracle (lowest-OOD-error) candidate over "
        rf"{runs[0]['n_trials']} trials $\times$ {len(runs)} seeds ({seeds}) "
        rf"$={P['n']}$ draws per cell. "
        rf"The geometric floor $\sqrt{{3}}\,\eps_W/r={floor:.3f}$ of \Cref{{cor:nearboundary}} "
        r"(poly-7 binding, poly-9 gives "
        rf"{geo['poly9_ols']['floor_l2']:.3f}) is marked; "
        rf"entries above the $5$-fold CV baseline ${cv_ref:.3f}$ are bold. "
        r"Block~(ii): the same sweep re-run over "
        rf"${len(widths)}$~widths $\times$ ${len(nvals)}$~segment sizes $\times$ "
        rf"${n_sig_coll}$~noise levels $\sigma \in \{{{sig_list}\}}$ "
        rf"$={len(cells)}$ cells, binned by the Prop.~\ref{{prop:strip}} statistic "
        r"$t_V=\sqrt{n_V}\,\norm{\Delta}_{L^2(\mathcal{D}_V)}/\sigma$ alone "
        r"($\Delta$ measured per cell against the hardest wrong candidate). "
        r"$\#w$, $\#n_V$, $\#\sigma$ count the distinct values pooled in each bin: "
        r"cells sharing a $t_V$ agree to within "
        rf"{max_sd:.3f} however they are built. "
        rf"All {len(above)} cells with $t_V \ge {t_rule:g}$ beat cross-validation; "
        rf"of the {len(below)} below, {n_nobeat} do not."
    )

    col_spec = "lrrrrr"
    head = (r"\multicolumn{6}{l}{\textbf{(i)} Accuracy vs.\ segment width "
            r"and size ($\sigma=10^{-2}$)}")
    out = table_header(cap, "tab:exp31_width_sweep", col_spec, head)

    out += ("    $w$ & $w/|W|$ (\\%) & "
            + " & ".join(rf"$n_V{{=}}{v}$" for v in nvals) + " \\\\\n")
    out += "    \\cmidrule(lr){1-6}\n"

    marked = False
    for w in widths:
        if not marked and w > floor:
            out += (r"    \multicolumn{6}{l}{\quad\itshape "
                    rf"$\sqrt{{3}}\,\eps_W/r = {floor:.3f}$: \Cref{{cor:nearboundary}} "
                    r"geometric floor} \\" "\n")
            marked = True
        cells_row = []
        for v in nvals:
            a = P["width_acc"][_ck(SIGMA_REF, REGIME_REF, w, v)]
            s = f"{a:.3f}"
            cells_row.append(f"\\textbf{{{s}}}" if a > cv_ref else s)
        out += (f"    {w:.2f} & {100.0 * w / win:.2f} & "
                + " & ".join(cells_row) + " \\\\\n")

    out += "    \\cmidrule(lr){1-6}\n"
    out += (r"    $5$-fold CV & --- & \multicolumn{4}{c}{"
            rf"{cv_ref:.3f}" r"} \\" "\n")

    out += "    \\midrule\n"
    out += ("    \\multicolumn{6}{l}{\\textbf{(ii)} The same "
            f"{len(cells)} cells binned by $t_V$ alone" "} \\\\\n")
    out += "    \\cmidrule(lr){1-6}\n"
    out += (r"    $t_V$ & cells & $\#w$ & $\#n_V$ & $\#\sigma$ & "
            r"accuracy \\" "\n")
    out += "    \\cmidrule(lr){1-6}\n"
    for b in bins:
        if b["lo"] == 0:
            rng_s = rf"$<{b['hi']:g}$"
        elif np.isinf(b["hi"]):
            rng_s = rf"$\ge {b['lo']:g}$"
        else:
            rng_s = rf"${b['lo']:g}$--${b['hi']:g}$"
        out += (f"    {rng_s} & {b['n']} & {b['n_w']} & {b['n_nv']} & "
                f"{b['n_sig']} & {b['acc']:.3f} $\\pm$ {b['sd']:.3f} \\\\\n")

    out += table_footer()
    return out


# %% build — CLI / shell entry point
def build(runs=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = render(runs)
    if tex is None:
        return None
    out.write_text(tex)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    tex = render()
    if tex:
        print(tex)
