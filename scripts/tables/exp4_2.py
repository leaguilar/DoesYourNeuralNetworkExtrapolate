#!/usr/bin/env python3
"""Exp 4.2 (Tor-Cov): coverage at full interaction order, ID/OOD R^2 over (d, M).

Per-cell counterpart to the coverage panel of \\Cref{fig:torus_scope} (panel
(c)). Exact $\\varphi$ held fixed; target is $g_{d,q}$ at $q=d$ (full
interaction order); coverage is varied through $M$ at $N=M^d$. Cells with
$N$ above the budget are dropped (status != "ok").

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
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
    _latex_thousands,
    fmt,
    table_footer,
    table_header,
)

EID = "4.2"
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp4_2.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp4_2] no input found, skipping")
        return None

    by = defaultdict(lambda: {"id_r2": [], "ood_r2": [], "ood_rmse": []})
    for run in runs:
        for cell in run["cells"].values():
            if cell.get("status") != "ok":
                continue
            key = (cell["d"], cell["M"])
            by[key]["id_r2"].append(cell["id_r2"])
            by[key]["ood_r2"].append(cell["ood_r2"])
            by[key]["ood_rmse"].append(cell["ood_rmse"])
    keys = sorted(by.keys())

    n_seeds = len(runs)
    out = []
    out.append(table_header(
        caption=(
            r"Torus benchmark, condition (c). Exact $\varphi$ held fixed at "
            r"full interaction order $q{=}d$; the only knob is per-coordinate "
            r"coverage $M$ at $N{=}M^d$. OOD $R^2$ rises with $M$ once the "
            r"sample budget covers each torus cell, locating the coverage "
            r"threshold $M_{\mathrm{crit}} \approx 4$ (\Cref{sec:scope}). "
            r"Cells with $N$ above budget are omitted. "
            r"Best OOD $R^2 \geq 0.95$ per row group in \textbf{bold}; "
            r"mean$\pm$std over " + f"{n_seeds}" + r" seeds."
        ),
        label="tab:exp42",
        col_spec="rrrrrr",
        header_row=(r"$d$ & $M$ & $N$ & ID $R^2$ & OOD $R^2$ & OOD RMSE"),
    ))
    last_d = None
    for (d, M) in keys:
        if last_d is not None and d != last_d:
            out.append(r"    \midrule")
        last_d = d
        N = M ** d
        id_m  = float(np.mean(by[(d, M)]["id_r2"]))
        id_s  = float(np.std(by[(d, M)]["id_r2"]))
        ood_m = float(np.mean(by[(d, M)]["ood_r2"]))
        ood_s = float(np.std(by[(d, M)]["ood_r2"]))
        rmse_m = float(np.mean(by[(d, M)]["ood_rmse"]))
        rmse_s = float(np.std(by[(d, M)]["ood_rmse"]))
        id_str   = fmt(id_m,   id_s,   bold=False,           decimals=3)
        ood_str  = fmt(ood_m,  ood_s,  bold=(ood_m >= 0.95), decimals=3)
        rmse_str = fmt(rmse_m, rmse_s, bold=False,           decimals=3)
        out.append(
            f"    {d} & {M} & {_latex_thousands(N)} & "
            f"{id_str} & {ood_str} & {rmse_str} \\\\"
        )
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
