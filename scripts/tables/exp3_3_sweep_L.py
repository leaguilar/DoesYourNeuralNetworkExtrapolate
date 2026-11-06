#!/usr/bin/env python3
"""Sweep L: learned-PE variants at baseline (L=4P, L_ood=2P, ds=64).

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

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp3_3_sweepL.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    return load_exp3_3_results(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp3_3_sweep_L] no input found, skipping")
        return None

    runs = [r for r in runs if r.get("sweep_tag") == "L"]
    if not runs:
        return ""
    periods = sorted({r["period"] for r in runs})
    pairs = [(m, pe) for m in MODEL_ORDER_ARCHINV
             for pe in ("learned_fourier_1freq", "learned_fourier_kfreq",
                        "learnable_sinusoidal")]
    cap = (r"Sweep L: learned-PE variants at baseline "
           r"($L_\mathrm{train}{=}4P$, $L_\mathrm{OOD}{=}2P$, "
           r"$d_\mathrm{state}{=}64$); tests whether SGD recovers the correct "
           r"frequency from random init and matches \textbf{exact Fourier}.")
    return _archinv_sweep_table(runs, "period", "P", periods, pairs,
                                 cap, "tab:exp33_sweepL")


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
