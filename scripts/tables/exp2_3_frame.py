#!/usr/bin/env python3
"""RW3 v2 hero table: PE × backbone × {ID, OOD-chrom, OOD-length} Pearson r.

Column "Length-OOD" is the load-bearing one — train at $W=200$, evaluate at
$W=500$. Right $\varphi$ extrapolates; learned PE collapses; pretrained
DNA FMs fail without injection.

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

from _loaders import load_exp2_3_full_results
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
from _exp2_3_common import (
    BACKBONE_DISPLAY_CDS,
    PE_DISPLAY_CDS,
    PE_ORDER_CDS_FM,
    PE_ORDER_CDS_TINY,
)

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp2_3_frame.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    return load_exp2_3_full_results(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_3_frame] no input found, skipping")
        return None

    if not runs:
        return ""

    by_key = defaultdict(list)  # (backbone, pe) -> [run]
    for r in runs:
        by_key[(r["backbone"], r["pe_name"])].append(r)
    n_seeds = max(len(v) for v in by_key.values())

    splits = [("r_test",        r"ID test"),
              ("r_ood_chrom",   r"OOD chr"),
              ("r_ood_length",  r"Length-OOD")]
    n_splits = len(splits)

    cap = (
        r"Exp.~2.3 (CDS-frame, yeast): Pearson $r$ between predicted and "
        r"observed codon-frame ($P{=}3$) label, by PE $\varphi$ and backbone. "
        r"Length-OOD trains at $W{=}200$, evaluates at $W{=}500$. "
        r"\textbf{Codon $P{=}3$} is the oracle. Pretrained DNA FMs are at "
        r"chance without explicit codon injection; the random-init Tiny "
        r"transformer with the right $\varphi$ wins outright. "
        rf"Mean $\pm$ std over {n_seeds} seeds; best per backbone in bold."
    )

    col_spec = "l" + "r" * n_splits
    head = (r"\textbf{$\varphi$} & "
            + " & ".join(rf"\textbf{{{label}}}" for _, label in splits))

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{cap}}}\n  \\label{{tab:exp23_frame}}\n"
           f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
           f"    {head} \\\\\n    \\midrule\n")

    blocks = [
        ("tiny_transformer", PE_ORDER_CDS_TINY),
        ("hyenadna",         PE_ORDER_CDS_FM),
        ("caduceus",         PE_ORDER_CDS_FM),
    ]
    for bi, (backbone, pe_order) in enumerate(blocks):
        rows_present = any((backbone, pe) in by_key for pe in pe_order)
        if not rows_present:
            continue
        out += (f"    \\multicolumn{{{1 + n_splits}}}{{l}}"
                f"{{{BACKBONE_DISPLAY_CDS[backbone]}}} \\\\\n")

        # Best per split for this backbone
        best_per_split = {}
        for split_key, _ in splits:
            vals = []
            for pe in pe_order:
                rs = by_key.get((backbone, pe))
                if not rs:
                    continue
                vs = [r.get(split_key) for r in rs]
                vs = [v for v in vs if v is not None]
                if vs:
                    vals.append(float(np.mean(vs)))
            best_per_split[split_key] = max(vals) if vals else None

        for pe in pe_order:
            rs = by_key.get((backbone, pe))
            if not rs:
                continue
            cells = []
            for split_key, _ in splits:
                vals = [r.get(split_key) for r in rs if r.get(split_key) is not None]
                if not vals:
                    cells.append("---")
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals))
                bold = (best_per_split[split_key] is not None
                        and abs(m - best_per_split[split_key]) < 1e-3)
                cell = f"{m:.2f} $\\pm$ {s:.2f}"
                cells.append(f"\\textbf{{{cell}}}" if bold else cell)
            out += (f"    {PE_DISPLAY_CDS.get(pe, pe)} & "
                    + " & ".join(cells) + " \\\\\n")
        if bi < len(blocks) - 1:
            out += "    \\midrule\n"
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
