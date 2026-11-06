#!/usr/bin/env python3
"""Top CDS-specific periodicities of the AT/GC track (W=200 windows).

The CDS / non-CDS power ratio is the load-bearing column: a period whose
spectral mass is enriched inside CDS regions is a candidate that the
model could exploit; a period that is equally strong everywhere carries
no CDS-discriminating information.

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

from _loaders import load_exp2_3_fft
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

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp2_3_fft.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    return load_exp2_3_fft(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(summary=None):
    if summary is None:
        summary = _load_input()
    if not summary:
        print(f"  [table_exp2_3_fft] no input found, skipping")
        return None

    if not summary:
        return ""

    ratio_peaks = summary["peaks"]["cds_over_non"]
    # New format (fft_periodicity.py >= 2026-04): each ratio peak carries
    # cds_power / non_power / ratio. Older summary.json fall back to lookup.
    rows = []
    for r in ratio_peaks:
        period = r["period_bp"]
        cds_p = r.get("cds_power")
        non_p = r.get("non_power")
        ratio = r.get("ratio", r.get("power"))
        rows.append((period, cds_p, non_p, ratio))

    n = summary.get("n_cds_windows")
    cap = (
        rf"FFT of yeast plus-strand AT-content "
        rf"({summary['annotation_source']}, $W{{=}}{summary['window']}$ bp, "
        rf"$n{{=}}{_latex_thousands(n)}$ CDS windows). Top-{len(rows)} periods "
        r"by CDS/non-CDS power ratio; bold = codon band ($P\in[2.6, 4.1]$). "
        r"Only $P{=}3$ is enriched inside CDS; the ${\sim}10.4$ bp helical "
        r"peak is not, justifying the codon-only $\varphi^\star$ in "
        r"\Cref{tab:exp23_frame}."
    )

    col_spec = "rrrr"
    head = (r"\textbf{period (bp)} & "
            r"\textbf{CDS power} & "
            r"\textbf{non-CDS power} & "
            r"\textbf{ratio (CDS / non)}")

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{cap}}}\n  \\label{{tab:exp23_fft}}\n"
           f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
           f"    {head} \\\\\n    \\midrule\n")
    for period, cds_p, non_p, ratio in rows:
        # Bold rows in the codon band (2.6-4.1 bp); they're the load-bearing finding.
        is_codon = 2.6 <= period <= 4.1
        bo, bc = (r"\textbf{", r"}") if is_codon else ("", "")
        cds_str = f"{cds_p:.2f}" if cds_p is not None else "---"
        non_str = f"{non_p:.2f}" if non_p is not None else "---"
        out += (f"    {bo}{period:.3f}{bc} & "
                f"{bo}{cds_str}{bc} & "
                f"{bo}{non_str}{bc} & "
                f"{bo}{ratio:.2f}{bc} \\\\\n")
    out += table_footer()
    return out


# %% build — CLI / shell entry point
def build(summary=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = render(summary)
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
