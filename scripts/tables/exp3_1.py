#!/usr/bin/env python3
"""Exp 3.1 (selection protocol): near-boundary validation vs CV.

Selection counts are broken out into a multicolumn group (one column per
library transform) so the per-transform mass is visible at a glance. The
canonically-correct \texttt{fourier\_ols} header is bolded; the
near-boundary row is shaded.

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _loaders import load_results
from _fmt import (
    CONDITION_LABELS,
    _fmt_with_thousands,
    _latex_thousands,
    agg,
    best_bold,
    fmt,
    lbl,
    table_footer,
    table_header,
)

EID = '3.1'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp3_1.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp3_1] no input found, skipping")
        return None

    cv_accs  = [r["results"]["cv_accuracy"]  * 100 for r in runs]
    val_accs = [r["results"]["val_accuracy"] * 100 for r in runs]
    k = runs[0]["config"]["K"]
    n_trials = runs[0]["n_trials"]
    rand_base = 100.0 / k

    cv_m,  cv_s  = agg(cv_accs)
    val_m, val_s = agg(val_accs)

    # Selection counts: aggregate across runs (sum per transform).
    transforms = ["fourier_ols", "poly7_ols", "poly9_ols", "raw_ols"]
    cv_tot  = {t: 0 for t in transforms}
    val_tot = {t: 0 for t in transforms}
    for r in runs:
        for t in transforms:
            cv_tot[t]  += r["results"].get("cv_counts",  {}).get(t, 0)
            val_tot[t] += r["results"].get("val_counts", {}).get(t, 0)
    n_total = sum(cv_tot.values())  # same for val

    cap = (
        rf"Exp.~3.1: Selection protocol. {n_trials} trials, DGP $\sin(x)$, "
        rf"$K={k}$ candidate feature maps (all OLS; random baseline "
        rf"{rand_base:.0f}\%). "
        r"\colorbox{gray!12}{\strut Shaded row} is the near-boundary protocol "
        r"(\Cref{cor:nearboundary}), querying the $\eps_W$-divergence "
        r"boundary outside the agreement zone. Selection counts are over "
        rf"{n_total} (trial $\times$ seed) draws; the correct "
        r"$\varphi(x){=}(\sin x, \cos x)$ (bold header) maximises the count. "
        r"Accuracy (\%), mean $\pm$ std; best in bold."
    )

    tf_disp = {
        "fourier_ols": r"\textbf{Fourier}",
        "poly7_ols":   r"Poly-7",
        "poly9_ols":   r"Poly-9",
        "raw_ols":     r"Raw $x$",
    }
    n_tf = len(transforms)
    col_spec = "lr" + "r" * n_tf
    head_top = (r"\textbf{Criterion} & \textbf{Accuracy (\%)} & "
                rf"\multicolumn{{{n_tf}}}{{c}}{{\textbf{{Selection counts}}}}")
    head_sub = " & & " + " & ".join(tf_disp[t] for t in transforms)
    cmid     = rf"\cmidrule(lr){{3-{2 + n_tf}}}"

    # Hand-build header so we can splice in the multicolumn + cmidrule.
    out = (
        f"\\begin{{table}}[ht]\n"
        f"  \\centering\n"
        f"  \\small\n"
        f"  \\caption{{{cap}}}\n"
        f"  \\label{{tab:exp31}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        f"    \\toprule\n"
        f"    {head_top} \\\\\n"
        f"    {cmid}\n"
        f"    {head_sub} \\\\\n"
        f"    \\midrule\n"
    )

    def cell(t, count, totals):
        bold = (t == "fourier_ols" and count == max(totals.values()))
        return f"\\textbf{{{count}}}" if bold else f"{count}"

    cv_cells  = " & ".join(cell(t, cv_tot[t],  cv_tot)  for t in transforms)
    val_cells = " & ".join(cell(t, val_tot[t], val_tot) for t in transforms)

    out += f"    5-fold CV & {fmt(cv_m, cv_s)} & {cv_cells} \\\\\n"
    out += (f"    \\rowcolor{{gray!12}}\n"
            f"    Near-boundary val. & {fmt(val_m, val_s, bold=True)} & "
            f"{val_cells} \\\\\n")
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
