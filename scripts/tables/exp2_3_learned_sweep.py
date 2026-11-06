#!/usr/bin/env python3
"""Init-period sweep for `learned_fourier_k1`: SGD does not move log_freq.

Columns: init_P, mean(final_P), mean(|Δlog_freq|), r_id, r_oodL.

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

from _loaders import load_exp2_3_learned_sweep
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

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp2_3_rashomon_init.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    return load_exp2_3_learned_sweep(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_3_learned_sweep] no input found, skipping")
        return None

    if not runs:
        return ""

    by_init = defaultdict(list)
    for r in runs:
        # Bucket by requested init period to handle FP rounding.
        key = round(float(r["init_period_request"]), 3)
        by_init[key].append(r)

    n_seeds = max(len(v) for v in by_init.values())

    cap = (
        r"Init-period sweep for \texttt{learned\_fourier\_k1} on CDS-frame "
        r"($W{=}200$, $20$ epochs). $|\Delta \log\omega|<0.01$ everywhere: "
        r"SGD does not move the frequency. Only \textbf{rational multiples "
        r"of $3$} ($P_0\in\{1.5, 3, 6\}$) reach length-OOD~$\geq 0.85$, so "
        r"the Rashomon set is unreachable from random init. "
        rf"Mean $\pm$ std over {n_seeds} seeds."
    )

    col_spec = "rrrrr"
    head = (r"\textbf{init $P$} & \textbf{final $P$} & "
            r"\textbf{$|\Delta\log\omega|$} & "
            r"\textbf{$r_\mathrm{ID}$} & \textbf{$r_\mathrm{OOD\text{-}len}$}")

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{cap}}}\n  \\label{{tab:exp23_rashomon_init}}\n"
           f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
           f"    {head} \\\\\n    \\midrule\n")

    HARMONICS = {1.5, 3.0, 6.0}

    for init_p in sorted(by_init.keys()):
        rs = by_init[init_p]
        finals = np.array([r["final_period"] for r in rs])
        moves  = np.array([r["log_freq_movement"] for r in rs])
        rids   = np.array([r["r_id"] for r in rs])
        rools  = np.array([r["r_ood_length"] for r in rs])
        is_harmonic = init_p in HARMONICS
        bold_open  = r"\textbf{" if is_harmonic else ""
        bold_close = r"}"        if is_harmonic else ""
        out += (
            f"    {bold_open}{init_p:.2f}{bold_close} & "
            f"{finals.mean():.3f} & "
            f"{moves.mean():.4f} & "
            f"{rids.mean():+.2f} $\\pm$ {rids.std():.2f} & "
            f"{bold_open}{rools.mean():+.2f} $\\pm$ {rools.std():.2f}{bold_close}"
            f" \\\\\n"
        )
    out += table_footer()
    return out


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
