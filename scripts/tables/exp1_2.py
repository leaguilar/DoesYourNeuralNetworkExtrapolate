#!/usr/bin/env python3
"""Exp 1.2 — DGP specificity, full table mirroring Exp 1.1 layout.

6-column layout: DGP | Feature map | Model | ID% | OOD P1 % | OOD P2 %
Per DGP: three feature-map subgroups separated by \cmidrule.
  • Correct feature map (shaded): OLS, MLP, SINDy
  • Wrong feature map:            OLS, MLP, SINDy
  • No feature map:               MLP, SINDy
OOD P1 / OOD P2 evaluate the *same* predictions against $x^2$ and
$e^{\alpha(x-1)}$ targets respectively on $[2, 10]$.

`tfm_runs` is accepted for back-compat with TABLE_BUILDERS_TFM but ignored.

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

EID = '1.2'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp1_2.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp1_2] no input found, skipping")
        return None

    import math

    SHADE = r"\rowcolor{gray!12}"
    CMIDRULE = "    \\cmidrule(lr){2-6}\n"

    dgps    = ["P1_powerlaw", "P2_exponential"]
    dgp_tex = {"P1_powerlaw":    r"$P_1$: $x^2$",
               "P2_exponential": r"$P_2$: $e^{\alpha(x-1)}$"}

    # Per-DGP layout: ordered (feature_map_label, model_label, condition_key, shaded).
    # log-log is correct for P1; log-y is correct for P2.
    layout = {
        "P1_powerlaw": [
            ("log-log",      "OLS",     "log_log_ols",     True),
            ("log-log",      "MLP",     "log_log_mlp",     True),
            ("log-log",      "SINDy",   "sindy_loglog",    True),
            ("log-log",      "TabPFN",  "tabpfn_loglog",   True),
            ("log-log",      "TimesFM", "timesfm_loglog",  True),
            ("log-$y$",      "OLS",     "log_y_ols",       False),
            ("log-$y$",      "MLP",     "log_y_mlp",       False),
            ("log-$y$",      "SINDy",   "sindy_logy",      False),
            ("log-$y$",      "TabPFN",  "tabpfn_logy",     False),
            ("log-$y$",      "TimesFM", "timesfm_logy",    False),
            ("None",         "MLP",     "raw_mlp",         False),
            ("None",         "SINDy",   "sindy_raw",       False),
            ("None",         "TabPFN",  "tabpfn_raw",      False),
            ("None",         "TimesFM", "timesfm",         False),
        ],
        "P2_exponential": [
            ("log-$y$",      "OLS",     "log_y_ols",       True),
            ("log-$y$",      "MLP",     "log_y_mlp",       True),
            ("log-$y$",      "SINDy",   "sindy_logy",      True),
            ("log-$y$",      "TabPFN",  "tabpfn_logy",     True),
            ("log-$y$",      "TimesFM", "timesfm_logy",    True),
            ("log-log",      "OLS",     "log_log_ols",     False),
            ("log-log",      "MLP",     "log_log_mlp",     False),
            ("log-log",      "SINDy",   "sindy_loglog",    False),
            ("log-log",      "TabPFN",  "tabpfn_loglog",   False),
            ("log-log",      "TimesFM", "timesfm_loglog",  False),
            ("None",         "MLP",     "raw_mlp",         False),
            ("None",         "SINDy",   "sindy_raw",       False),
            ("None",         "TabPFN",  "tabpfn_raw",      False),
            ("None",         "TimesFM", "timesfm",         False),
        ],
    }

    def _vals(d, c, metric):
        out = []
        for run in runs:
            v = run.get("dgps", {}).get(d, {}).get(c, {}).get(metric)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out.append(float(v))
        return out

    def _agg(vals):
        if not vals:
            return None
        return float(np.mean(vals)), float(np.std(vals))

    # Global best per OOD column (across both DGP blocks), matching Exp 1.1.
    all_p1_means, all_p2_means = [], []
    for d in dgps:
        for _, _, ck, _ in layout[d]:
            t1 = _agg(_vals(d, ck, "ood_error_vs_p1"))
            t2 = _agg(_vals(d, ck, "ood_error_vs_p2"))
            all_p1_means.append(t1[0] if t1 is not None else float("inf"))
            all_p2_means.append(t2[0] if t2 is not None else float("inf"))
    best_p1 = min(all_p1_means)
    best_p2 = min(all_p2_means)

    n = len(runs)
    cap = (
        r"Exp.~1.2: DGP specificity. $P_1(x){=}x^2$ vs.\ "
        r"$P_2(x){=}e^{(\ln 4)(x-1)}$ agree on $[1, 2]$ and diverge by "
        r"${\sim}10^5$ on $[2, 10]$. The DGP column identifies the training-data source. "
        r"OOD $P_1$ and OOD $P_2$ score the same predictions against $P_1$ and "
        r"$P_2$ as ground truth, respectively. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the correct feature map "
        r"for OLS. TabPFN / TimesFM are zero-shot; the feature map is applied "
        r"to inputs and (where applicable) outputs. OOD relative error (\%), "
        r"mean $\pm$ std over "
        f"{n}" r" seeds; best per column in bold."
    )
    out = table_header(
        cap, "tab:exp12", "lllrrr",
        r"\textbf{DGP} & \textbf{Feature map} & \textbf{Model} & "
        r"\textbf{ID (\%)} & \textbf{OOD $P_1$ (\%)} & \textbf{OOD $P_2$ (\%)}"
    )

    def _cell(tup, best):
        if tup is None:
            return "--"
        return fmt(*tup, bold=abs(tup[0] - best) < 1e-6)

    for di, d in enumerate(dgps):
        rows = layout[d]
        prev_fm = None
        for i, (fm, mdl, ck, shaded) in enumerate(rows):
            if prev_fm is not None and fm != prev_fm:
                out += CMIDRULE
            prev_fm = fm

            id_t = _agg(_vals(d, ck, "id_error"))
            p1_t = _agg(_vals(d, ck, "ood_error_vs_p1"))
            p2_t = _agg(_vals(d, ck, "ood_error_vs_p2"))

            id_str = "--" if id_t is None else fmt(*id_t)
            dgp_str = dgp_tex[d] if i == 0 else ""
            prefix  = f"    {SHADE}\n    " if shaded else "    "

            out += (f"{prefix}{dgp_str} & {fm} & {mdl} & "
                    f"{id_str} & {_cell(p1_t, best_p1)} & "
                    f"{_cell(p2_t, best_p2)} \\\\\n")

        if di != len(dgps) - 1:
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
