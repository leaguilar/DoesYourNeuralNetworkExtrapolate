#!/usr/bin/env python3
"""Wider window doesn't rescue learned_fourier_k1 from a wrong init (P=5).

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

from _loaders import load_exp2_3_window_sweep
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

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp2_3_window_sweep.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    return load_exp2_3_window_sweep(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_3_window_sweep] no input found, skipping")
        return None

    if not runs:
        return ""

    by_w = defaultdict(list)
    for r in runs:
        by_w[int(r["window"])].append(r)

    n_seeds = max(len(v) for v in by_w.values())
    init_p = float(runs[0]["init_period"])

    cap = (
        rf"Window-size sweep at wrong init period ($P_0{{=}}{init_p:g}$) for "
        r"\texttt{learned\_fourier\_k1}. Gradient magnitude grows with $W$ "
        r"but $|\Delta\log\omega|$ does not, and ID stays at chance: a larger "
        rf"window does not force SGD onto $P{{=}}3$. Mean over {n_seeds} seeds."
    )

    col_spec = "rrrrr"
    head = (r"\textbf{$W$} & "
            r"\textbf{final $P$} & "
            r"\textbf{$|\Delta\log\omega|$} & "
            r"\textbf{$\langle |\partial_{\log\omega}\mathcal{L}| \rangle$} & "
            r"\textbf{$r_\mathrm{ID}$}")

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{cap}}}\n  \\label{{tab:exp23_window_sweep}}\n"
           f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
           f"    {head} \\\\\n    \\midrule\n")

    for w in sorted(by_w.keys()):
        rs = by_w[w]
        finals = np.array([r["final_period"] for r in rs])
        moves  = np.array([r["log_freq_movement"] for r in rs])
        grads  = np.array([r["mean_grad_norm_log_freq"] for r in rs])
        rids   = np.array([r["r_id"] for r in rs])
        out += (
            f"    {w} & "
            f"{finals.mean():.3f} & "
            f"{moves.mean():.4f} & "
            f"{grads.mean():.2f} & "
            f"{rids.mean():+.2f} $\\pm$ {rids.std():.2f} \\\\\n"
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
