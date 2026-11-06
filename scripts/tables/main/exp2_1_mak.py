#!/usr/bin/env python3
"""Exp 2.1 MAK (excerpt): 4-column Feature map | Model | ID | OOD.

Bilinear-feature rows are shaded (canonically-correct commitment).

Run as a script:
    conda run -n data_analysis python scripts/main_make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import math
import sys
from pathlib import Path

import numpy as np

# Make sibling + parent modules importable in both shell-import and cell-mode
# contexts. The parent (scripts/tables) hosts _loaders, _fmt, etc.
_HERE = Path(__file__).resolve().parent
_TABLES = _HERE.parent
for _p in (_HERE, _TABLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
from _helpers import cond_metric, dgp_metric, fmt_or_dash

EID = '2.1'
OUT = Path(__file__).resolve().parents[3] / "tables" / "main" / "table_exp2_1_main.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_1_main] no input found, skipping")
        return None

    # Aliases used inside the legacy body.
    _dgp_metric = dgp_metric

    layout = [
        ("None",     "MLP", "raw_mlp", False),
        ("Bilinear", "OLS", "mak_ols", True),
        ("Bilinear", "MLP", "mak_mlp", True),
    ]

    cands = [c for _, _, c, _ in layout]
    bests = [cond_metric(runs, c, "ood_error") for c in cands]
    finite = [b[0] for b in bests if b is not None]
    best = min(finite) if finite else None

    rows = []
    prev_fm = None
    for fm, mdl, cond, shaded in layout:
        ide = cond_metric(runs, cond, "id_error")
        ood = cond_metric(runs, cond, "ood_error")
        if ood is None:
            continue
        if prev_fm is not None and fm != prev_fm:
            rows.append("    \\cmidrule(lr){1-4}")
        prev_fm = fm
        bold = (best is not None and abs(ood[0] - best) < 1e-3)
        shade = "    \\rowcolor{gray!12}\n" if shaded else ""
        rows.append(
            f"{shade}    {fm} & {mdl} & "
            f"{fmt_or_dash(ide)} & {fmt_or_dash(ood, bold=bold)} \\\\"
        )

    cap = (
        r"Exp.~2.1 MAK (excerpt). 2-node mass-action system, training "
        r"$(x_1, x_2) \in [0.4, 1.2]^2$, OOD "
        r"$[0, 2.0]^2 \setminus [0.4, 1.2]^2$. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the canonically-correct "
        r"bilinear feature map $[1, x_i, x_i x_j]$. "
        r"Mean $\pm$ std over 3 seeds; best OOD in bold; full grid in "
        r"App.~\ref{app:exp21_full}."
    )

    return (
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        f"  \\caption{{{cap}}}\n  \\label{{tab:exp21_main}}\n"
        "  \\begin{tabular}{llrr}\n    \\toprule\n"
        "    \\textbf{Feature map} & \\textbf{Model} & "
        "\\textbf{ID (\\%)} & \\textbf{OOD (\\%)} \\\\\n    \\midrule\n"
        + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    )


# %% build — CLI / shell entry point
def build(runs=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = render(runs)
    if not tex:
        return None
    out.write_text(tex)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    tex = render()
    if tex:
        print(tex)
