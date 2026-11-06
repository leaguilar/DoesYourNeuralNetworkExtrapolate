#!/usr/bin/env python3
"""Sweep B: P=64 fixed, varied L_train (long-context advantage).

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
from _exp3_3_archinv import (
    MODEL_DISPLAY as MODEL_DISPLAY_ARCHINV,
    MODEL_ORDER as MODEL_ORDER_ARCHINV,
    PE_DISPLAY as PE_DISPLAY_ARCHINV,
    PE_ORDER as PE_ORDER_ARCHINV,
    sweep_table as _archinv_sweep_table,
)

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp3_3_sweepB.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    return load_exp3_3_results(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp3_3_sweep_B] no input found, skipping")
        return None

    runs = [r for r in runs if r.get("sweep_tag") == "B"]
    if not runs:
        return ""
    L_vals = sorted({r["L_train"] for r in runs})
    pairs = [(m, pe) for m in MODEL_ORDER_ARCHINV
             for pe in ("exact_fourier", "learned_ape", "no_pe")]
    cap = (r"Sweep B: long-context regime ($P{=}64$, $L_\mathrm{train}$ "
           r"varied); checks that the PE-driven OOD pattern survives larger "
           r"contexts.")
    return _archinv_sweep_table(runs, "L_train", r"L_\mathrm{train}",
                                 L_vals, pairs,
                                 cap, "tab:exp33_sweepB")


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
