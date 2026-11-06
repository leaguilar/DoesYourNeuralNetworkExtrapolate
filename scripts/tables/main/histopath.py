#!/usr/bin/env python3
"""Exp 2.4 Camelyon17 (excerpt): from-scratch OOD AUC for the five input maps.

Compact 3-column cut of `tab:histopath` (Input map | Role | OOD AUC). Only the
from-scratch ResNet-18 arm exists; the full grid with ID/OOD accuracy is the
appendix table.

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

from _loaders import load_histopath
from _fmt import agg, fmt, table_footer, table_header
from _helpers import fmt_or_dash
from histopath import ARM_LABELS, ARM_ORDER, PHI_ARMS, _scratch_stats

# Main-text excerpt: the raw baseline and the two adaptive input maps only.
# Controls (colorctrl, raw+caug) are in the appendix table.
EXCERPT_ARMS = ["raw", "stainnorm", "channorm"]

OUT = Path(__file__).resolve().parents[3] / "tables" / "main" / "table_histopath_main.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"

SHADE = r"\rowcolor{gray!12}"

ARM_ROLES = {
    "raw":       r"no preprocessing",
    "stainnorm": r"$\varphi$: Macenko stain normalisation",
    "channorm":  r"adaptive, stain-agnostic",
    "colorctrl": r"fixed recolouring (negative control)",
    "raw+caug":  r"colour-jitter augmentation (DG baseline)",
}


def _load_input(results_dir=None):
    return load_histopath(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs or not runs.get("scratch"):
        print("  [table_histopath_main] no input found, skipping")
        return None

    scratch = _scratch_stats(runs["scratch"])
    if not scratch:
        print("  [table_histopath_main] no scratch arms found, skipping")
        return None

    n_seeds = max(c["n_seeds"] for c in scratch.values())
    min_id_auc = min(c["id_auc"][0] for c in scratch.values())
    best = max(c["ood_auc"][0] for c in scratch.values())

    body = []
    for arm in EXCERPT_ARMS:
        if arm not in scratch:
            continue
        c = scratch[arm]
        is_best = abs(c["ood_auc"][0] - best) < 1e-9
        shade = f"    {SHADE}\n" if arm in PHI_ARMS else ""
        body.append(
            f"{shade}    {ARM_LABELS[arm]} & {ARM_ROLES[arm]} & "
            f"{fmt(*c['ood_auc'], bold=is_best, decimals=4)} \\\\"
        )

    cap = (
        r"Exp.~2.4 Camelyon17 (excerpt). ResNet-18 trained from scratch on "
        r"centres $\{0,\,3,\,4\}$, evaluated on held-out centre $2$, at the "
        r"\emph{ID-selected} epoch so selection never uses OOD labels. "
        rf"ID AUC is $\geq {min_id_auc:.3f}$ everywhere, so the whole spread "
        r"is out-of-distribution. \colorbox{gray!12}{\strut Shaded rows} are "
        r"adaptive input maps $\varphi$: \texttt{stainnorm} aligns the stain "
        r"basis, \texttt{channorm} only per-image channel statistics, and both "
        rf"carry the commitment. Mean $\pm$ std over {n_seeds} seeds; best in "
        r"bold. Controls (a fixed non-adaptive recolouring, and colour "
        r"augmentation) and the full grid are in \Cref{tab:histopath}."
    )

    return (
        "\\begin{table}[tb]\n  \\centering\n  \\small\n"
        f"  \\caption{{{cap}}}\n  \\label{{tab:histopath_main}}\n"
        "  \\begin{tabular}{llr}\n    \\toprule\n"
        "    \\textbf{Input map} & \\textbf{Role} & \\textbf{OOD AUC} \\\\\n"
        "    \\midrule\n"
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
    print(render())
