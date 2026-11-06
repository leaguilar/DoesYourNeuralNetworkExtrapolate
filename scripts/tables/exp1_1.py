#!/usr/bin/env python3
"""Exp 1.1 — Observational equivalence: sin(x) vs. Taylor-9.

6-column layout: DGP | Feature map | Model | ID% | OOD P1 (%) | OOD P2 (%)

P1 block — Fourier feature map (shaded): OLS, SINDy, MLP, TimesFM(cov), TabPFN
P1 block — No feature map:               SINDy, MLP, TabPFN, TimesFM
P2 block — No feature map:               SINDy, MLP, TabPFN, TimesFM

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

EID = '1.1'
# Preserve the three-seed selection used in the manuscript.
PAPER_SEEDS = frozenset({42, 43, 44})
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp1_1.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    runs = [run for run in runs if run.get("seed") in PAPER_SEEDS]
    if not runs:
        print(f"  [table_exp1_1] no input found, skipping")
        return None

    import math

    SHADE = r"\rowcolor{gray!12}"
    CMIDRULE = "    \\cmidrule(lr){2-6}\n"

    def _scalar(cond_key, metric):
        vals = []
        for r in runs:
            v = r.get("conditions", {}).get(cond_key)
            if v is None:
                continue
            x = v.get(metric)
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            vals.append(float(x))
        if not vals:
            return None
        return float(np.mean(vals)), float(np.std(vals))

    def _id(cond_key):
        vals = []
        for r in runs:
            v = r.get("conditions", {}).get(cond_key)
            if v is None:
                continue
            ie = v.get("id_error")
            if ie is None or (isinstance(ie, float) and math.isnan(ie)):
                continue
            vals.append(float(ie))
        if not vals:
            return None
        return float(np.mean(vals)), float(np.std(vals))

    def _cell(tup, best):
        if tup is None:
            return "--"
        m, s = tup
        bold = abs(m - best) < 1e-3
        return fmt(m, s, bold=bold)

    def _fmtrow(dgp_str, fmap_str, model_str, id_tup,
                p1_tup, best_p1, p2_tup, best_p2, shaded=False):
        prefix = f"    {SHADE}\n    " if shaded else "    "
        id_str = "--" if id_tup is None else fmt(*id_tup)
        return (f"{prefix}{dgp_str} & {fmap_str} & {model_str} & "
                f"{id_str} & {_cell(p1_tup, best_p1)} & {_cell(p2_tup, best_p2)} \\\\\n")

    n = len(runs)

    # Global best per column (across all blocks so both ends of the cross are bolded)
    p1_committed_p1 = ["fourier_ols", "fourier_mlp",
                        "timesfm_fourier_cov", "tabpfn_fourier",
                        "sindy_fourier", "raw_mlp", "tabpfn", "timesfm"]
    p2_committed_p1 = ["sindy_p2ctx", "raw_mlp_p2ctx", "tabpfn_p2ctx", "timesfm_p2ctx"]

    all_p1_vals = (
        [(_scalar(ck, "ood_error")       or (float("inf"), 0))[0] for ck in p1_committed_p1]
      + [(_scalar(ck, "ood_error_vs_p1") or (float("inf"), 0))[0] for ck in p2_committed_p1]
    )
    best_p1 = min(all_p1_vals)

    all_p2_vals = (
        [(_scalar(ck, "ood_error_vs_p2") or (float("inf"), 0))[0] for ck in p1_committed_p1]
      + [(_scalar(ck, "ood_error")       or (float("inf"), 0))[0] for ck in p2_committed_p1]
    )
    best_p2 = min(all_p2_vals)

    rel_diff = float(np.mean([
        r["indistinguishability"]["train_max_rel_diff_pct"] for r in runs
    ]))

    cap = (
        r"Exp.~1.1: $P_1{=}\sin(x)$ vs.\ $P_2{=}\mathrm{Taylor}_9(x)$, train "
        r"$[-\pi, \pi]$ (max rel.\ dev.\ "
        rf"${{<}}{rel_diff:.1f}\%$"
        r"), OOD $[\pi, 3\pi]$. The DGP column identifies the training-data source. "
        r"OOD $P_1$ and OOD $P_2$ score the same predictions against $P_1$ and "
        r"$P_2$ as ground truth, respectively. \colorbox{gray!12}{\strut Shaded rows} use a "
        r"feature map. OOD relative error (\%), mean $\pm$ std over "
        f"{n}" r" seeds; best per column in bold."
    )

    out = table_header(
        cap, "tab:exp11", "lllrrr",
        r"\textbf{DGP} & \textbf{Feature map} & \textbf{Model} & "
        r"\textbf{ID (\%)} & \textbf{OOD $P_1$ (\%)} & \textbf{OOD $P_2$ (\%)}"
    )

    # ── P1 block: structured models (shaded) ─────────────────────────────────
    # Fourier feature map rows + SINDy (poly-9 library), all shaded together to
    # contrast with the black-box rows below the cmidrule.
    dgp_p1 = r"$P_1$: $\sin(x)$"
    for dgp_s, fm_s, mdl, ck in [
        (dgp_p1, "Fourier", "OLS",           "fourier_ols"),
        ("",     "Fourier", "SINDy",         "sindy_fourier"),
        ("",     "Fourier", "MLP",           "fourier_mlp"),
        ("",     "Fourier", "TabPFN",        "tabpfn_fourier"),
        ("",     "Fourier", "TimesFM (cov)", "timesfm_fourier_cov"),
    ]:
        out += _fmtrow(dgp_s, fm_s, mdl,
                       _id(ck),
                       _scalar(ck, "ood_error"),       best_p1,
                       _scalar(ck, "ood_error_vs_p2"), best_p2,
                       shaded=True)
    out += CMIDRULE

    # ── P1 block: black-box models (unshaded) ────────────────────────────────
    for dgp_s, fm_s, mdl, ck in [
        ("", "None", "SINDy",         "sindy_raw"),
        ("", "None", "MLP",     "raw_mlp"),
        ("", "None", "TabPFN",  "tabpfn"),
        ("", "None", "TimesFM", "timesfm"),
    ]:
        out += _fmtrow(dgp_s, fm_s, mdl,
                       _id(ck),
                       _scalar(ck, "ood_error"),       best_p1,
                       _scalar(ck, "ood_error_vs_p2"), best_p2,
                       shaded=False)
    out += "    \\midrule\n"

    # ── P2-committed block ────────────────────────────────────────────────────
    # All rows trained on P2 data. ID = error vs P2 targets on training domain.
    # OOD P1 col = ood_error_vs_p1; OOD P2 col = ood_error.
    dgp_p2 = r"$P_2$: Taylor-9"
    for dgp_s, fm_s, mdl, ck in [
        (dgp_p2, "None", "SINDy",   "sindy_p2ctx"),
        ("",     "None", "MLP",     "raw_mlp_p2ctx"),
        ("",     "None", "TabPFN",  "tabpfn_p2ctx"),
        ("",     "None", "TimesFM", "timesfm_p2ctx"),
    ]:
        out += _fmtrow(dgp_s, fm_s, mdl,
                       _id(ck),
                       _scalar(ck, "ood_error_vs_p1"), best_p1,
                       _scalar(ck, "ood_error"),        best_p2)

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
