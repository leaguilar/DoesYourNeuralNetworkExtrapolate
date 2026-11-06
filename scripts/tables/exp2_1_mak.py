#!/usr/bin/env python3
"""Exp 2.1 (MAK): vector field, ODE feature engineering.

4-column layout: Feature map | Model | ID% | OOD% (Exp 1.2 style).
Bilinear-feature rows are shaded as the canonically-correct commitment.

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

EID = '2.1'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp2_1.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_1_mak] no input found, skipping")
        return None

    SHADE = r"\rowcolor{gray!12}"
    CMIDRULE = "    \\cmidrule(lr){1-4}\n"

    # (feature_map_label, model_label, condition_key, shaded)
    layout = [
        ("None",      "MLP", "raw_mlp", False),
        ("Bilinear",  "OLS", "mak_ols", True),
        ("Bilinear",  "MLP", "mak_mlp", True),
    ]

    data = {c: {"id": [], "ood": []} for _, _, c, _ in layout}
    for run in runs:
        for _, _, c, _ in layout:
            data[c]["id"].append(run["conditions"][c]["id_error"])
            data[c]["ood"].append(run["conditions"][c]["ood_error"])

    n    = len(runs)
    best = min(np.mean(data[c]["ood"]) for _, _, c, _ in layout)
    cfg  = runs[0]["config"]

    lo = cfg["train_center"] - cfg["train_half"]
    hi = cfg["train_center"] + cfg["train_half"]
    cap = (
        r"Exp.~2.1 (MAK): vector field on a canonical synthetic 2-node graph "
        r"\citep{vasiliauskaite2024generalization}. "
        r"True process: $\dot{x}_i = F - Bx_i - Rx_ix_j$ "
        rf"with textbook parameter values $F={cfg['F']},\,B={cfg['B']},\,R={cfg['R']}$ "
        r"(dimensionless, representative of the CSTR mass-action regime: $F$ is "
        r"constant inflow, $B$ first-order outflow, $R$ bimolecular rate). "
        r"Training: $(x_1, x_2) \in ["
        rf"{lo:.1f}, {hi:.1f}]^2$; "
        r"OOD: $[0, "
        rf"{cfg['state_max']}]^2 \setminus [{lo:.1f}, {hi:.1f}]^2$. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the canonically-correct "
        r"bilinear feature map $[1, x_i, x_i x_j]$, which linearises the "
        r"target globally: OLS on these features recovers $(F, -B, -R)$ exactly. "
        r"OOD relative error (\%), mean $\pm$ std over "
        f"{n} seeds. Best OOD in bold."
    )
    out = table_header(
        cap, "tab:exp21", "llrr",
        r"\textbf{Feature map} & \textbf{Model} & "
        r"\textbf{ID (\%)} & \textbf{OOD (\%)}"
    )

    prev_fm = None
    for fm, mdl, c, shaded in layout:
        if prev_fm is not None and fm != prev_fm:
            out += CMIDRULE
        prev_fm = fm

        id_m,  id_s  = agg(data[c]["id"])
        ood_m, ood_s = agg(data[c]["ood"])
        is_best = abs(ood_m - best) < 1e-6
        prefix = f"    {SHADE}\n    " if shaded else "    "
        out += (f"{prefix}{fm} & {mdl} & {fmt(id_m, id_s)} & "
                f"{fmt(ood_m, ood_s, bold=is_best)} \\\\\n")

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
