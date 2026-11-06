#!/usr/bin/env python3
"""Exp 4.1 (Tor-Repr): RMSE per d for raw / learned-Fourier / exact-phi.

Reproduces selected fixed-budget rows of Nagler v2c Table 1 (the
'interaction' target rows).

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

EID = '4.1'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp4_1.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp4_1] no input found, skipping")
        return None

    by_d = defaultdict(lambda: {"raw_mlp": [], "learned_fourier_mlp": [], "exact_phi_mlp": []})
    for run in runs:
        for cell in run["conditions"].values():
            d = cell["d"]
            for cond in ["raw_mlp", "learned_fourier_mlp", "exact_phi_mlp"]:
                by_d[d][cond].append(cell[cond]["ood_rmse"])
    ds = sorted(by_d.keys())

    n_seeds = len(runs)
    out = []
    out.append(table_header(
        caption=(
            r"Torus benchmark, condition (a). Fixed budget $N=4096$ on the "
            r"$d$-torus with target $g_\mathrm{int}$. The exact map "
            r"$\varphi(x)=(\sin x_j, \cos x_j)$ dominates raw and learned-Fourier "
            r"baselines. OOD RMSE, mean$\pm$std over " + f"{n_seeds}" + r" seeds; "
            r"best per row in \textbf{bold}."
        ),
        label="tab:exp41",
        col_spec="lrrr",
        header_row=(r"$d$ & raw MLP & learned-Fourier MLP & exact-$\varphi$ MLP"),
    ))
    for d in ds:
        means = {}
        stds  = {}
        for cond in ("raw_mlp", "learned_fourier_mlp", "exact_phi_mlp"):
            v = by_d[d][cond]
            if v:
                means[cond] = float(np.mean(v))
                stds[cond]  = float(np.std(v))
            else:
                means[cond] = float("nan")
                stds[cond]  = float("nan")
        best = min((m for m in means.values() if not np.isnan(m)), default=float("inf"))
        cells = []
        for cond in ("raw_mlp", "learned_fourier_mlp", "exact_phi_mlp"):
            is_best = (means[cond] == best)
            cells.append(fmt(means[cond], stds[cond], bold=is_best, decimals=3))
        out.append(f"    {d} & " + " & ".join(cells) + r" \\")
    out.append(table_footer())
    return "\n".join(out) + "\n"


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
