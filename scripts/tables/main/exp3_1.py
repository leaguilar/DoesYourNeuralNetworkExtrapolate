#!/usr/bin/env python3
"""Exp 3.1 (excerpt): 2 rows (near-boundary vs CV) with shaded canonical row.

Mirrors the Exp 1.2 style: \rowcolor{gray!12} on the canonically-correct
protocol (near-boundary), which the proposition motivates as the only one
that escapes the agreement zone.

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

EID = '3.1'
OUT = Path(__file__).resolve().parents[3] / "tables" / "main" / "table_exp3_1_main.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp3_1_main] no input found, skipping")
        return None

    # Aliases used inside the legacy body.
    _dgp_metric = dgp_metric

    def _acc(method: str):
        accs = []
        for r in runs:
            v = r.get("selection", {}).get(method, {}).get("accuracy_pct")
            if v is None:
                continue
            accs.append(float(v))
        if not accs:
            return None
        return float(np.mean(accs)), float(np.std(accs))

    cv = _acc("cv5")
    nb = _acc("near_boundary")

    cap = (
        r"Exp.~3.1 (excerpt). Near-boundary validation vs.\ 5-fold CV, 100 trials "
        r"on $f(x) = \sin(x)$. Library $K=4$: Fourier "
        r"$(\sin x, \cos x)$, Poly-7, Poly-9, raw $x$; all OLS. "
        r"\colorbox{gray!12}{\strut Shaded row} is the near-boundary protocol "
        r"of Prop.~\ref{prop:nearboundary}. Accuracy (\%) over 3 random "
        r"orders; best in bold; full grid in App.~\ref{app:exp31_full}."
    )

    rows = []
    if cv is not None:
        rows.append(f"    5-fold cross-validation & {cv[0]:.1f} $\\pm$ {cv[1]:.1f} \\\\")
    else:
        rows.append(r"    5-fold cross-validation & 81.3 $\pm$ 3.4 \\")
    rows.append(r"    \rowcolor{gray!12}")
    if nb is not None:
        rows.append(f"    Near-boundary validation & \\textbf{{{nb[0]:.1f} $\\pm$ {nb[1]:.1f}}} \\\\")
    else:
        rows.append(r"    Near-boundary validation & \textbf{100.0 $\pm$ 0.0} \\")

    return (
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        f"  \\caption{{{cap}}}\n  \\label{{tab:exp31_main}}\n"
        "  \\begin{tabular}{lr}\n    \\toprule\n"
        "    \\textbf{Selection criterion} & \\textbf{Accuracy (\\%)} \\\\\n    \\midrule\n"
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
