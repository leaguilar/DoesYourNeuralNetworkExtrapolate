#!/usr/bin/env python3
"""Exp 3.3 Architectural invariance (excerpt): passthrough of the compact builder.

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

from _loaders import load_exp3_3_results
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

OUT = Path(__file__).resolve().parents[3] / "tables" / "main" / "table_exp3_3_main.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"


def _load_input(results_dir=None):
    return load_exp3_3_results(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp3_3_main] no input found, skipping")
        return None

    # Aliases used inside the legacy body.
    _dgp_metric = dgp_metric

    from make_tables import table_exp3_3  # legacy ad-hoc builder  # full builder
    full = table_exp3_3(runs)
    if not full:
        return ""
    return full.replace(r"\label{tab:exp33}", r"\label{tab:exp33_main}")


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
