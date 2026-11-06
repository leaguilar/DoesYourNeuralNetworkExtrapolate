#!/usr/bin/env python3
"""Exp 2.2 (Kepler): ablation [Variable subset] x [Log transform] x [Model class].

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

EID = '2.2'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp2_2.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp2_2_kepler] no input found, skipping")
        return None

    import math

    n = len(runs)
    n_train = runs[0]["config"]["n_train"]
    n_ood   = runs[0]["config"]["n_ood"]
    a_split = runs[0]["config"]["a_split"]
    a_ood   = runs[0]["config"]["a_ood_max"]

    def _agg(cond, key="ood_error"):
        vals = []
        for r in runs:
            v = r["results"].get(cond, {}).get(key, float("nan"))
            if isinstance(v, float) and not math.isnan(v):
                vals.append(v)
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    def _fmt(m, s, bold=False):
        if math.isnan(m):
            return r"\textit{---}"
        if m > 99999:
            return r"$\gg 9{,}999$"
        f = f"{_fmt_with_thousands(m, 1)} $\\pm$ {_fmt_with_thousands(s, 1)}"
        return f"\\textbf{{{f}}}" if bold else f

    # Recover Kepler exponents from the first run's OLS coef
    coef_aM = runs[0]["results"].get("log_aM_ols", {}).get("coef", [None, None])
    b_a_str = f"{coef_aM[0]:.3f}" if coef_aM[0] is not None else "?"
    b_M_str = f"{coef_aM[1]:.3f}" if len(coef_aM) > 1 and coef_aM[1] is not None else "?"
    sindy_k = runs[0]["results"].get("log_aM_sindy", {}).get("n_terms", "?")

    cap = (
        r"Exp.~2.2 (Kepler): third law on NASA exoplanets. "
        rf"$n_\mathrm{{train}}={_latex_thousands(n_train)}$ ($a<{a_split}$~AU), "
        rf"$n_\mathrm{{OOD}}={_latex_thousands(n_ood)}$ ($\leq{a_ood:.0f}$~AU). "
        r"OOD rel.\ error (\%), mean $\pm$ std over "
        f"{n} seeds; best OOD per variable group in bold. "
        r"Correct OLS recovers $b_a=" + b_a_str
        + r"$, $b_{M_{\mathrm{star}}}=" + b_M_str + r"$ (theory $1.500, -0.500$); "
        rf"SINDy on the correct map gives $k={sindy_k}$ active terms. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the correct feature map."
    )

    SHADE    = r"\rowcolor{gray!12}"
    CMIDRULE = "    \\cmidrule(lr){2-5}\n"

    out = table_header(
        cap, "tab:exp22", "lllrr",
        r"\textbf{Variables} & \textbf{Feature map} & \textbf{Model} & "
        r"\textbf{ID (\%)} & \textbf{OOD (\%)}"
    )

    # Structure: list of (group_label, subgroups)
    # Each subgroup: (feature_map_label, correct, [(cond_key, model_label)])
    # correct=True → rows are shaded gray; best OOD per group is bolded
    groups = [
        # ── Group 1: {a} ───────────────────────────────────────────────────
        (r"$a$", [
            (r"$a$",     False, [
                ("raw_ols",      "OLS"),
                ("raw_sindy",    "SINDy"),
                ("raw_a_mlp",   "MLP"),
                ("raw_a_tabpfn","TabPFN"),
            ]),
            (r"$\log a$", False, [
                ("log_a_ols",     "OLS"),
                ("log_a_sindy",   "SINDy"),
                ("log_a_mlp",     "MLP"),
                ("log_a_tabpfn",  "TabPFN"),
            ]),
        ]),
        # ── Group 2: {a, M★} ───────────────────────────────────────────────
        (r"$a,\,M_{\mathrm{star}}$", [
            (r"$a,\,M_{\mathrm{star}}$", False, [
                ("raw_aM_ols",    "OLS"),
                ("raw_aM_sindy",  "SINDy"),
                ("raw_aM_mlp",    "MLP"),
                ("raw_aM_tabpfn", "TabPFN"),
            ]),
            (r"$\log a,\,M_{\mathrm{star}}$", False, [
                ("log_a_linM_ols",    "OLS"),
                ("log_a_linM_sindy",  "SINDy"),
                ("log_a_linM_mlp",    "MLP"),
                ("log_a_linM_tabpfn", "TabPFN"),
            ]),
            (r"$a,\,\log M_{\mathrm{star}}$", False, [
                ("lina_logM_ols",    "OLS"),
                ("lina_logM_sindy",  "SINDy"),
                ("lina_logM_mlp",    "MLP"),
                ("lina_logM_tabpfn", "TabPFN"),
            ]),
            (r"$\log a,\,\log M_{\mathrm{star}}$", True, [   # CORRECT — Kepler's law
                ("log_aM_ols",    "OLS"),
                ("log_aM_sindy",  "SINDy"),
                ("log_aM_mlp",    "MLP"),
                ("log_aM_tabpfn", "TabPFN"),
            ]),
        ]),
        # ── Group 3a: {a, R★} ──────────────────────────────────────────────
        (r"$a,\,R_{\mathrm{star}}$", [
            (r"$\log a,\,\log R_{\mathrm{star}}$", False, [
                ("log_aR_ols",    "OLS"),
                ("log_aR_sindy",  "SINDy"),
                ("log_aR_mlp",    "MLP"),
                ("log_aR_tabpfn", "TabPFN"),
            ]),
        ]),
        # ── Group 3b: {a, T_eff} ───────────────────────────────────────────
        (r"$a,\,T_\mathrm{eff}$", [
            (r"$\log a,\,\log T_\mathrm{eff}$", False, [
                ("log_aTe_ols",    "OLS"),
                ("log_aTe_sindy",  "SINDy"),
                ("log_aTe_mlp",    "MLP"),
                ("log_aTe_tabpfn", "TabPFN"),
            ]),
        ]),
        # ── Group 4: {a, M★, R★, T_eff} ───────────────────────────────────
        (r"$a,\,M_{\mathrm{star}},\,R_{\mathrm{star}},\,T_\mathrm{eff}$", [
            (r"$\log a,\,\log M_{\mathrm{star}},\,\log R_{\mathrm{star}},\,\log T_\mathrm{eff}$", False, [
                ("log_aMRTe_ols",    "OLS"),
                ("log_aMRTe_sindy",  "SINDy"),
                ("log_aMRTe_mlp",    "MLP"),
                ("log_aMRTe_tabpfn", "TabPFN"),
            ]),
        ]),
    ]

    for gi, (group_label, subgroups) in enumerate(groups):
        # Best OOD across the entire variable group
        all_cond_keys = [ck for _, _c, conds in subgroups for ck, _ in conds]
        ood_vals = [_agg(ck, "ood_error")[0] for ck in all_cond_keys]
        finite_oods = [v for v in ood_vals if not math.isnan(v) and v < 99999]
        best = min(finite_oods) if finite_oods else float("nan")

        first_in_group = True
        for si, (fm_label, correct, conds) in enumerate(subgroups):
            first_in_fm = True
            for ci, (cond, model_lbl) in enumerate(conds):
                vars_str = group_label if first_in_group else ""
                fm_str   = fm_label   if first_in_fm   else ""
                id_m,  id_s  = _agg(cond, "id_error")
                ood_m, ood_s = _agg(cond, "ood_error")
                is_best = (not math.isnan(ood_m)) and (ood_m < 99999) and abs(ood_m - best) < 1e-3
                prefix = f"    {SHADE}\n    " if correct else "    "
                out += (f"{prefix}{vars_str} & {fm_str} & {model_lbl} & "
                        f"{_fmt(id_m, id_s)} & {_fmt(ood_m, ood_s, bold=is_best)} \\\\\n")
                first_in_group = False
                first_in_fm    = False
            # Thin rule between feature-map sub-groups within same variable group
            if si < len(subgroups) - 1:
                out += CMIDRULE

        if gi < len(groups) - 1:
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
