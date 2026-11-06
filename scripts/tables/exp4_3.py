#!/usr/bin/env python3
"""Exp 4.3 (Tor-Cap): capacity at varied coverage, ID/OOD R^2 over (d, M, q).

Reproduces Nagler v2c Table 2.

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

EID = '4.3'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp4_3.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp4_3] no input found, skipping")
        return None

    by = defaultdict(lambda: {"id_r2": [], "ood_r2": []})
    for run in runs:
        for cell in run["cells"].values():
            if cell.get("status") != "ok":
                continue
            key = (cell["d"], cell["M"], cell["q"])
            by[key]["id_r2"].append(cell["id_r2"])
            by[key]["ood_r2"].append(cell["ood_r2"])
    keys = sorted(by.keys())

    n_seeds = len(runs)
    out = []
    out.append(table_header(
        caption=(
            r"Torus benchmark, condition (b). Exact $\varphi$ held fixed; "
            r"target $g_{d,q}$ has interaction order $q$. OOD $R^2$ collapses "
            r"as $q$ approaches $d$, even when ID $R^2\!\approx\!1$, exposing a "
            r"capacity bottleneck at finite coverage. Best OOD $R^2 \geq 0.95$ "
            r"per row group in \textbf{bold}; $N=M^d$. Mean$\pm$std over "
            + f"{n_seeds}" + r" seeds."
        ),
        label="tab:exp43",
        col_spec="rrrrr",
        header_row=(r"$d$ & $M$ & $q$ & ID $R^2$ & OOD $R^2$"),
    ))
    last_d = None
    for (d, M, q) in keys:
        if last_d is not None and d != last_d:
            out.append(r"    \midrule")
        last_d = d
        id_m  = float(np.mean(by[(d,M,q)]["id_r2"]))
        id_s  = float(np.std(by[(d,M,q)]["id_r2"]))
        ood_m = float(np.mean(by[(d,M,q)]["ood_r2"]))
        ood_s = float(np.std(by[(d,M,q)]["ood_r2"]))
        id_str  = fmt(id_m,  id_s,  bold=False,           decimals=3)
        ood_str = fmt(ood_m, ood_s, bold=(ood_m >= 0.95), decimals=3)
        out.append(f"    {d} & {M} & {q} & {id_str} & {ood_str} \\\\")
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
