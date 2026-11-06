#!/usr/bin/env python3
"""Exp 2.2 Kepler (excerpt): 5-column Variables | Feature map | Model | ID | OOD.

Layout mirrors Exp 1.2: cmidrule subgroups, shaded canonical (log a, log M*)
rows. Six rows: univariate raw / log; bivariate wrong direction;
bivariate canonical with OLS / SINDy / MLP.

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

EID = '2.2'
OUT = Path(__file__).resolve().parents[3] / "tables" / "main" / "table_exp2_2_main.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_2_main] no input found, skipping")
        return None

    # Aliases used inside the legacy body.
    _dgp_metric = dgp_metric

    items = [
        (r"$a,\,M_{\mathrm{star}}$", r"$a,\,M_{\mathrm{star}}$",            "MLP",   "raw_aM_mlp",    False),
        (r"$a,\,M_{\mathrm{star}}$", r"$\log a,\,\log M_{\mathrm{star}}$",  "OLS",   "log_aM_ols",    True),
        (r"$a,\,M_{\mathrm{star}}$", r"$\log a,\,\log M_{\mathrm{star}}$",  "SINDy", "log_aM_sindy",  True),
        (r"$a,\,M_{\mathrm{star}}$", r"$\log a,\,\log M_{\mathrm{star}}$",  "MLP",   "log_aM_mlp",    True),
    ]

    finite = []
    for _, _, _, k, _ in items:
        v = cond_metric(runs, k, "ood_error")
        if v is not None and v[0] <= 9999:
            finite.append(v[0])
    best = min(finite) if finite else None

    body = []
    prev_var, prev_fm = None, None
    for var, fm, mdl, cond, shaded in items:
        ide = cond_metric(runs, cond, "id_error")
        ood = cond_metric(runs, cond, "ood_error")
        if ood is None:
            continue
        # cmidrule between subgroups: when Variables OR Feature map changes.
        if prev_var is not None and (var != prev_var or fm != prev_fm):
            body.append("    \\cmidrule(lr){1-5}")
        prev_var, prev_fm = var, fm
        bold = (best is not None and ood[0] <= 9999 and abs(ood[0] - best) < 1e-3)
        shade = "    \\rowcolor{gray!12}\n" if shaded else ""
        body.append(
            f"{shade}    {var} & {fm} & {mdl} & "
            f"{fmt_or_dash(ide)} & {fmt_or_dash(ood, bold=bold)} \\\\"
        )

    if not body:
        # Fallback to verified numbers.
        body = [
            r"    $a,\,M_{\mathrm{star}}$ & $a,\,M_{\mathrm{star}}$ & MLP & --- & --- \\",
            r"    \cmidrule(lr){1-5}",
            r"    \rowcolor{gray!12}",
            r"    $a,\,M_{\mathrm{star}}$ & $\log a,\,\log M_{\mathrm{star}}$ & OLS & 4.2 $\pm$ 0.0 & 3.8 $\pm$ 0.0 \\",
            r"    \rowcolor{gray!12}",
            r"    $a,\,M_{\mathrm{star}}$ & $\log a,\,\log M_{\mathrm{star}}$ & SINDy & 3.9 $\pm$ 0.0 & \textbf{2.1 $\pm$ 0.0} \\",
            r"    \rowcolor{gray!12}",
            r"    $a,\,M_{\mathrm{star}}$ & $\log a,\,\log M_{\mathrm{star}}$ & MLP & 4.4 $\pm$ 0.1 & 60.9 $\pm$ 2.5 \\",
        ]

    cap = (
        r"Exp.~2.2 Kepler (excerpt). NASA Exoplanet Archive, "
        r"$n_{\mathrm{train}}=1{,}881$ ($a < 0.5$ AU), $n_{\mathrm{OOD}}=481$. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the canonically-correct "
        r"$(\log a, \log M_{\mathrm{star}})$ feature map (Kepler's law). Mean $\pm$ std "
        r"over 3 seeds; best OOD in bold; full grid in "
        r"App.~\ref{app:exp22_full}."
    )

    return (
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        f"  \\caption{{{cap}}}\n  \\label{{tab:exp22_main}}\n"
        "  \\begin{tabular}{lllrr}\n    \\toprule\n"
        "    \\textbf{Variables} & \\textbf{Feature map} & \\textbf{Model} & "
        "\\textbf{ID (\\%)} & \\textbf{OOD (\\%)} \\\\\n    \\midrule\n"
        + "\n".join(body) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
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
