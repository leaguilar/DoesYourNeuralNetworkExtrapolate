#!/usr/bin/env python3
"""Generate LaTeX tables from experiment results — dispatch shell.

Active table builders live in `scripts/tables/*.py` (one file per
artefact) and are wired through DISPATCH below. The legacy ad-hoc builders
(table_exp1_3, table_exp1_3_sindy, table_exp2_2, table_exp2_3, table_exp2_5,
table_exp2_6, table_exp1_3_manifold, table_exp3_1, table_summary, table_hero,
table_exp3_3, table_exp2_3_rw3_legacy, table_exp2_3_species_clean, table_hcs_rashomon,
_table_exp_rw1_old, table_exprw1_sindy_delta) are kept later in this file for
ad-hoc / appendix-supplement use; if you re-enable any of them, register them
in DISPATCH so sync_paper.py picks up the output.

Usage (from project root):
    conda run -n data_analysis python scripts/make_tables.py
    conda run -n data_analysis python scripts/make_tables.py --summary_only
    conda run -n data_analysis python scripts/make_tables.py --results_dir results --output_dir tables

Intentional orphans (results that exist on disk but aren't built into a TABLE_BUILDERS entry):
    - results/exp1_3/exp1_3__seed*.json: G1/G2/I1/I2 conditions for the
      Fourier and log-polynomial families. The G1 fourier_ols / fourier_mlp
      OOD numbers (~82% / 2.5%) are cited inline in paper/sections/appendix.tex
      (App. A.5 narrative). The shared SINDy-regime view of these DGPs lives
      in `tab:exp13sindy` / `tab:sindy_delta_consolidated`, built from
      `exp1_3_sindy/`. Keep `exp1_3/` until either the inline citation is
      replaced with a dedicated table or the prose moves the numbers into an
      existing builder.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Importing as `python scripts/make_tables.py` puts scripts on
# sys.path so the tables/ subpackage resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Active-artefact entry points.
from tables.exp1_1 import build as build_exp1_1
from tables.exp1_2 import build as build_exp1_2
from tables.exp3_1 import build as build_exp3_1
from tables.exp3_1_width_sweep import build as build_exp3_1_width_sweep
from tables.exp2_2_kepler import build as build_exp2_2_kepler
from tables.exp2_1_mak import build as build_exp2_1_mak
from tables.sindy_delta_consolidated import build as build_sindy_consolidated
from tables.exp4_1 import build as build_exp4_1
from tables.exp4_2 import build as build_exp4_2
from tables.exp4_3 import build as build_exp4_3
from tables.exp4_4 import build as build_exp4_4
from tables.exp4_4_ratio import build as build_exp4_4_ratio
from tables.exp3_3_sweep_A import build as build_exp3_3_A
from tables.exp3_3_sweep_B import build as build_exp3_3_B
from tables.exp3_3_sweep_C import build as build_exp3_3_C
from tables.exp3_3_sweep_D import build as build_exp3_3_D
from tables.exp3_3_sweep_L import build as build_exp3_3_L
from tables.exp2_3_frame import build as build_exp2_3_frame
from tables.exp2_3_learned_sweep import build as build_exp2_3_learned_sweep
from tables.exp2_3_fft import build as build_exp2_3_fft
from tables.exp2_3_window_sweep import build as build_exp2_3_window_sweep
from tables.histopath import build as build_histopath
from tables.e3_misspec import build as build_e3_misspec

# Re-export shared loaders + formatters so legacy ad-hoc builders below (and
# main_make_tables.py) keep working unchanged.
from tables._loaders import (
    load_exp2_3_fft,
    load_exp2_3_full_results,
    load_exp2_3_learned_sweep,
    load_exp2_3_species_clean,
    load_exp2_3_window_sweep,
    load_hero_results,
    load_histopath,
    load_exp3_3_results,
    load_results,
)
from tables._fmt import (
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
from tables._exp3_3_archinv import (
    MODEL_DISPLAY as MODEL_DISPLAY_ARCHINV,
    MODEL_ORDER as MODEL_ORDER_ARCHINV,
    PE_DISPLAY as PE_DISPLAY_ARCHINV,
    PE_ORDER as PE_ORDER_ARCHINV,
    sweep_table as _archinv_sweep_table,
)


# Active dispatch tables — each entry maps an experiment id to its build()
# function. Mirrors the pre-refactor TABLE_BUILDERS / TABLE_BUILDERS_TORUS
# layout and is consumed by main() below.
DISPATCH_BY_EID = {
    "1.1": build_exp1_1,
    "1.2": build_exp1_2,
    "2.1": build_exp2_1_mak,
    "2.2": build_exp2_2_kepler,
    "3.1": build_exp3_1,
    # Exp 3.1 near-boundary width sweep + t_V collapse (App. H). Its
    # experiment_id is the dotted id plus a suffix so it does not collide with
    # the selection-protocol results under the same 3.1 heading.
    "3.1_width_sweep": build_exp3_1_width_sweep,
    # Exp 4.5 — phi-misspecification sweep. Its per-seed files DO match the
    # exp*__<seed>__<ts>.json glob of load_results, so it dispatches by the
    # experiment_id written inside the JSON rather than by a dotted number.
    "E3_misspec": build_e3_misspec,
}
DISPATCH_TORUS = {
    "4.1": build_exp4_1,
    "4.2": build_exp4_2,
    "4.3": build_exp4_3,
    "4.4": build_exp4_4,
}
# Extra torus artefacts built from the runs of an eid that already has a
# builder in DISPATCH_TORUS (a dict cannot hold two builders under one key).
DISPATCH_TORUS_EXTRA = [
    ("4.4", build_exp4_4_ratio),
]
DISPATCH_HERO_SSM_SWEEPS = [
    ("A", build_exp3_3_A),
    ("B", build_exp3_3_B),
    ("C", build_exp3_3_C),
    ("D", build_exp3_3_D),
    ("L", build_exp3_3_L),
]


import numpy as np




# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------





def table_exp1_3(runs):
    """Exp 1.3 — SC gap diagnostic."""
    rho_train = float(np.mean([r["spearman"]["rho_sc_train"] for r in runs]))
    rho_gap   = float(np.mean([r["spearman"]["rho_sc_gap"]   for r in runs]))
    p_train   = float(np.mean([r["spearman"]["p_sc_train"]   for r in runs]))
    p_gap     = float(np.mean([r["spearman"]["p_sc_gap"]     for r in runs]))
    n_pairs   = runs[0]["spearman"]["n_pairs"]

    cap = (rf"Exp 1.3 SC gap diagnostic: Spearman $\rho$ between spectral complexity "
           rf"metrics and OOD error across {n_pairs} (DGP, transform) pairs "
           rf"({len(runs)} seeds).")
    out  = table_header(cap, "tab:exp13",
                        "lrrr",
                        r"\textbf{Metric} & $\boldsymbol{\rho}$ & $\boldsymbol{p}$-value & \textbf{Interpretation}")
    out += (f"    $\\mathrm{{SC}}_{{\\mathrm{{train}}}}$ & {rho_train:.3f} & "
            f"{p_train:.3e} & Weak predictor \\\\\n")
    out += (f"    $\\mathrm{{SC}}_{{\\mathrm{{gap}}}}$ & \\textbf{{{rho_gap:.3f}}} & "
            f"\\textbf{{{p_gap:.3e}}} & Strong predictor \\\\\n")
    out += table_footer()
    return out


def table_exp1_3_sindy(runs):
    """Exp 1.3 — SINDy coordinate diagnostic (replaces SC-gap table)."""
    import math

    # Aggregate across seeds
    dgps_agg = {}
    for run in runs:
        for dgp, transforms in run["records"].items():
            if dgp not in dgps_agg:
                dgps_agg[dgp] = {}
            for tf, v in transforms.items():
                if tf not in dgps_agg[dgp]:
                    dgps_agg[dgp][tf] = {
                        "correct": v["correct"],
                        "n_terms": [], "train_deriv_res": [], "ood_deriv_res": [],
                    }
                dgps_agg[dgp][tf]["n_terms"].append(v["n_terms"])
                dgps_agg[dgp][tf]["train_deriv_res"].append(v["train_deriv_res"])
                dgps_agg[dgp][tf]["ood_deriv_res"].append(v["ood_deriv_res"])

    DGP_LABELS = {
        "sin(x)":            r"$\sin(x)$",
        "sin(2x)":           r"$\sin(2x)$",
        "x^2":               r"$x^2$",
        "x^3+x":             r"$x^3+x$",
        "G1:sin(x)cos(2x)":  r"$\sin(x)\cos(2x)$",
        "G2:tanh(5sincos)":  r"$\tanh(5\sin(x)\cos(2x))$",
        r"I1:logy=2u-u²":    r"$\log y=2u-u^2$",
        "I1:logy=2u-u²":     r"$\log y=2u-u^2$",
        r"I2:logy=1-2e^{-x}": r"$\log y=1-2e^{-x}$",
    }
    # Handle the actual unicode key
    for k in list(dgps_agg.keys()):
        if "I1:" in k:
            DGP_LABELS[k] = r"$\log y=2u-u^2$"
        if "I2:" in k:
            DGP_LABELS[k] = r"$\log y=1-2e^{-x}$"

    TF_LABELS = {
        "raw":          r"Raw $x$",
        "fourier":      r"Fourier-2 (deg.\ 1)",
        "fourier2":     r"Fourier-2 (deg.\ 2)",
        "poly2":        r"Poly-2",
        "poly3":        r"Poly-3",
        "fourier-4":    r"Fourier-4 (deg.\ 2)",
        "fourier-6":    r"Fourier-6 (deg.\ 1)",
        "raw x->y":     r"Raw $x{\to}y$",
        "logx,raw y":   r"$\log x$, raw $y$",
        "loglog":       r"Log-log",
        "poly2 x->y":   r"Poly-2 $x{\to}y$",
        "logy,x-feat":  r"$\log y$, feat.\ $x$",
        "logy,v-state": r"$\log y$, state $v$",
        r"logy,e^-x":   r"$\log y$, feat.\ $e^{-x}$",
    }

    def fmt_res(v):
        if math.isnan(v):
            return r"\textit{---}"
        if v > 9999:
            return r"$\gg 9{,}999$"
        if v >= 100:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"

    rho  = float(np.mean([r["spearman"]["rho"]  for r in runs]))
    pval = float(np.mean([r["spearman"]["pval"] for r in runs]))
    n_pairs = runs[0]["spearman"]["n_pairs"]

    SHADE = r"\rowcolor{gray!12}"

    cap = (
        r"Exp.~1.3: SINDy coordinate diagnostic. "
        r"Each row pairs a DGP with a candidate feature map. "
        r"$k$: active SINDy terms after thresholding; "
        r"$\delta_\mathrm{tr}$: training derivative residual; "
        r"$\delta_\mathrm{OOD}$: OOD derivative residual. "
        r"Mean over " + f"{len(runs)}" + r" seeds. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the canonically-correct "
        r"coordinate; the bolded $\delta_\mathrm{OOD}$ value is $0$ up to "
        r"numerical precision. $\tanh(5\sin(x)\cos(2x))$ has no sparse "
        r"polynomial coordinate, so the diagnostic is non-zero on every "
        r"candidate. "
        rf"Spearman $\rho(\delta_\mathrm{{OOD}},\,\text{{wrong\,label}}) = {rho:.2f}$ "
        rf"($p={pval:.2e}$, $n={n_pairs}$ pairs)."
    )
    out = table_header(
        cap, "tab:exp13sindy", "llrrr",
        r"\textbf{DGP} & \textbf{Feature map} & $\boldsymbol{k}$ & "
        r"$\boldsymbol{\delta_\mathrm{tr}}$ & $\boldsymbol{\delta_\mathrm{OOD}}$"
    )

    dgp_order = [
        "sin(x)", "sin(2x)", "x^2", "x^3+x",
        "G1:sin(x)cos(2x)", "G2:tanh(5sincos)",
    ]
    # Append I1/I2 with their actual (potentially unicode) keys
    for k in dgps_agg:
        if "I1:" in k or "I2:" in k:
            dgp_order.append(k)

    n_groups = len([d for d in dgp_order if d in dgps_agg])
    written = 0
    for dgp in dgp_order:
        if dgp not in dgps_agg:
            continue
        written += 1
        transforms = dgps_agg[dgp]
        for tf, v in transforms.items():
            k_m   = float(np.mean(v["n_terms"]))
            tr_m  = float(np.mean(v["train_deriv_res"]))
            ood_m = float(np.mean(v["ood_deriv_res"]))
            proc_str = DGP_LABELS.get(dgp, dgp)
            tf_str   = TF_LABELS.get(tf, tf)
            k_str    = f"{k_m:.0f}"
            tr_str   = fmt_res(tr_m)
            ood_str  = fmt_res(ood_m)
            if v["correct"]:
                ood_str = f"\\textbf{{{ood_str}}}"
            prefix = f"    {SHADE}\n    " if v["correct"] else "    "
            out += (f"{prefix}{proc_str} & {tf_str} & {k_str} & "
                    f"{tr_str} & {ood_str} \\\\\n")
        if written < n_groups:
            out += "    \\midrule\n"

    out += table_footer()
    return out




def table_exp2_2(runs):
    """Exp 2.2 — ODE dynamics (SHO OLS conditions)."""
    conds = ["raw", "deriv", "poly2"]
    data  = {c: {"id": [], "ood": []} for c in conds}
    for run in runs:
        for c in conds:
            data[c]["id"].append(run["sho"][c]["id_error"])
            data[c]["ood"].append(run["sho"][c]["ood_error"])

    cfg = runs[0]["config"]
    n   = len(runs)
    best = min(np.mean(data[c]["ood"]) for c in conds)

    cap = (rf"Exp 2.2: ODE dynamics. Simple harmonic oscillator, "
           rf"$\omega_\mathrm{{train}}={cfg['omega_train']}$, "
           rf"$\omega_\mathrm{{OOD}}={cfg['omega_ood']}$ ($2\times$ shift). "
           rf"OLS conditions predict state increments. "
           rf"OOD relative error (\%), mean $\pm$ std "
           f"over {n} seeds. Best in bold.")
    out  = table_header(cap, "tab:legacy_old_exp22",
                        "lrr",
                        r"\textbf{Feature set} & \textbf{ID Error (\%)} & "
                        r"\textbf{OOD Error (\%)}")
    for c in conds:
        id_m,  id_s  = agg(data[c]["id"])
        ood_m, ood_s = agg(data[c]["ood"])
        is_best = abs(ood_m - best) < 1e-6
        out += f"    {lbl(c)} & {fmt(id_m, id_s)} & {fmt(ood_m, ood_s, bold=is_best)} \\\\\n"
    out += table_footer()
    return out


def table_exp2_3(runs):
    """Exp 2.3 — Wavelet."""
    conds = ["raw", "fft", "wavelet"]
    lbl_map = {"raw": r"\texttt{raw} (MLP)", "fft": r"\texttt{FFT} (MLP)",
               "wavelet": r"\texttt{Wavelet} (MLP)"}
    data  = {c: {"id": [], "ood": []} for c in conds}
    for run in runs:
        for c in conds:
            data[c]["id"].append(run["conditions"][c]["id_error"])
            data[c]["ood"].append(run["conditions"][c]["ood_error"])

    cfg = runs[0]["config"]
    n   = len(runs)
    best = min(np.mean(data[c]["ood"]) for c in conds)

    cap = (rf"Exp 2.3: Wavelet features. Training freq.\ {cfg['freq_train']} Hz, "
           rf"OOD freq.\ {cfg['freq_ood']} Hz. All conditions use MLP. "
           rf"OOD relative error (\%), mean $\pm$ std over {n} seeds. Best in bold.")
    out  = table_header(cap, "tab:legacy_old_exp23",
                        "lrr",
                        r"\textbf{Feature type} & \textbf{ID Error (\%)} & "
                        r"\textbf{OOD Error (\%)}")
    for c in conds:
        id_m,  id_s  = agg(data[c]["id"])
        ood_m, ood_s = agg(data[c]["ood"])
        is_best = abs(ood_m - best) < 1e-6
        out += f"    {lbl_map[c]} & {fmt(id_m, id_s)} & {fmt(ood_m, ood_s, bold=is_best)} \\\\\n"
    out += table_footer()
    return out




def table_exprw1_sindy_delta(runs):
    """Exp RW1: SINDy δ_ood diagnostic on Kepler's third law.

    Uses the same squared-NMSE δ formula as the toy battery
    (exp1_4_sindy_coord.py:127-130; logged by exprw1_kepler.py:_sindy_static).
    Demonstrates that the §5.2 diagnostic discriminates the canonical
    (log a, log M*) commitment from wrong feature maps on real exoplanet data.
    """
    cond_order = [
        ("log_aM_sindy",     r"$(\log a,\,\log M_{\mathrm{star}})$",                          "canonical"),
        ("log_aMRTe_sindy",  r"$(\log a,\,\log M,\,\log R,\,\log T_\mathrm{eff})$", "$+$ irrelevant"),
        ("log_aTe_sindy",    r"$(\log a,\,\log T_\mathrm{eff})$",                   r"$T_\mathrm{eff}$ proxy"),
        ("log_aR_sindy",     r"$(\log a,\,\log R_{\mathrm{star}})$",                          r"$R_{\mathrm{star}}$ proxy"),
        ("log_a_sindy",      r"$(\log a)$",                                          r"missing $M_{\mathrm{star}}$"),
        ("log_a_linM_sindy", r"$(\log a,\,M_{\mathrm{star}})$",                                "wrong $M$ log-side"),
        ("raw_aM_sindy",     r"$(a,\,M_{\mathrm{star}})$",                                     "no log"),
        ("lina_logM_sindy",  r"$(a,\,\log M_{\mathrm{star}})$",                                "wrong $a$ log-side"),
    ]
    agg = {c: {"k": [], "dtr": [], "dood": [], "ood": []} for c, _, _ in cond_order}
    for r in runs:
        for c, _, _ in cond_order:
            v = r["results"].get(c, {})
            agg[c]["k"].append(v.get("n_terms", float("nan")))
            agg[c]["dtr"].append(v.get("delta_train", float("nan")))
            agg[c]["dood"].append(v.get("delta_ood", float("nan")))
            agg[c]["ood"].append(v.get("ood_error_raw", v.get("ood_error", float("nan"))))

    def fmt_d(x):
        if not np.isfinite(x):
            return r"\textit{---}"
        if x >= 1e4:
            exp = int(np.floor(np.log10(x)))
            mant = x / 10 ** exp
            return f"${mant:.1f}{{\\times}}10^{{{exp}}}$"
        if x >= 100:
            return f"{x:.1f}"
        if x >= 1:
            return f"{x:.2f}"
        if x >= 0.001:
            return f"{x:.4f}"
        return f"{x:.2e}"

    def fmt_oe(x):
        if not np.isfinite(x):
            return r"\textit{nan}"
        if x > 9999:
            return r"$\gg 9{,}999$"
        return f"{x:.1f}"

    n_train = runs[0].get("config", {}).get("n_train", "?")
    n_ood   = runs[0].get("config", {}).get("n_ood",   "?")
    means = {c: float(np.mean(agg[c]["dood"])) for c, _, _ in cond_order}
    finite_means = [v for v in means.values() if np.isfinite(v)]
    best_dood = min(finite_means) if finite_means else float("nan")

    cap = (
        r"SINDy $\delta_\mathrm{OOD}$ diagnostic on Kepler's third law "
        rf"(real exoplanet data, $n_\mathrm{{train}}={_latex_thousands(n_train)}$, "
        rf"$n_\mathrm{{OOD}}={_latex_thousands(n_ood)}$). "
        r"The canonical $(\log a,\,\log M_{\mathrm{star}})$ representation achieves "
        r"$\delta_\mathrm{OOD} \approx 0$ as on the toy battery "
        r"(\S\ref{sec:exp13}); every wrong feature map yields "
        r"$\delta_\mathrm{OOD}$ orders of magnitude larger. "
        rf"Mean over {len(runs)} seeds. Best $\delta_\mathrm{{OOD}}$ in bold."
    )
    out = table_header(
        cap, "tab:legacy_kepler_sindy_delta", "llrrrr",
        r"\textbf{Feature map} & \textbf{Comment} & "
        r"$\boldsymbol{k}$ & "
        r"$\boldsymbol{\delta_\mathrm{tr}}$ & "
        r"$\boldsymbol{\delta_\mathrm{OOD}}$ & "
        r"\textbf{OOD err (\%)}"
    )

    for cond, label, comment in cond_order:
        a = agg[cond]
        k_m    = float(np.nanmean(a["k"]))
        dtr_m  = float(np.nanmean(a["dtr"]))
        dood_m = float(np.nanmean(a["dood"]))
        oe_m   = float(np.nanmean(a["ood"]))

        dood_str = fmt_d(dood_m)
        if np.isfinite(dood_m) and abs(dood_m - best_dood) < 1e-9:
            dood_str = f"\\textbf{{{dood_str}}}"

        out += (f"    {label} & {comment} & {k_m:.0f} & "
                f"{fmt_d(dtr_m)} & {dood_str} & {fmt_oe(oe_m)} \\\\\n")
    out += table_footer()
    return out














def _table_exp_rw1_old(runs):
    """Exp RW1 — Kepler's third law (real-world validation)."""
    conds = ["raw_mlp", "log_log_ols", "log_y_ols", "log_log_mlp", "log_y_mlp"]
    data  = {c: {"id": [], "ood": []} for c in conds}
    slopes = []
    for run in runs:
        for c in conds:
            data[c]["id"].append(run["results"][c]["id_error"])
            data[c]["ood"].append(run["results"][c]["ood_error"])
        slopes.append(run["results"]["log_log_ols"].get("slope", float("nan")))

    n = len(runs)
    slope_m = float(np.nanmean(slopes))
    best = min(np.mean(data[c]["ood"]) for c in conds)
    a_split   = runs[0]["config"]["a_split"]
    a_ood_max = runs[0]["config"]["a_ood_max"]
    n_train   = runs[0]["config"]["n_train"]
    n_ood     = runs[0]["config"]["n_ood"]

    cap = (rf"Exp.~2.2: Kepler's third law. {_latex_thousands(n_train)} exoplanets with $a<{a_split}$~AU (train), "
           rf"{_latex_thousands(n_ood)} with ${a_split}\leq a<{a_ood_max}$~AU (OOD). "
           rf"True DGP: $T \propto a^{{3/2}}$; fitted log-log slope $= {slope_m:.3f}$. "
           rf"Relative error (\%), mean $\pm$ std over {n} seeds. Best OOD in bold.")
    out = table_header(cap, "tab:legacy_old_exp22",
                       "lrr",
                       r"\textbf{Condition} & \textbf{ID Error (\%)} & "
                       r"\textbf{OOD Error (\%)}")
    for c in conds:
        id_m,  id_s  = agg(data[c]["id"])
        ood_m, ood_s = agg(data[c]["ood"])
        is_best = abs(ood_m - best) < 1e-6
        out += (f"    {lbl(c)} & {fmt(id_m, id_s)} & "
                f"{fmt(ood_m, ood_s, bold=is_best)} \\\\\n")
    out += table_footer()
    return out


def table_exp3_1(runs):
    """Exp 3.1 — Gating network."""
    conds = ["fourier", "poly2", "raw", "gating_fourier+poly2"]
    lbl_map = {
        "fourier":             r"\texttt{Fourier} (MLP single head)",
        "poly2":               r"\texttt{Poly-2} (MLP single head)",
        "raw":                 r"\texttt{Raw} (MLP single head)",
        "gating_fourier+poly2":r"\texttt{Gating} (Fourier + Poly-2)",
    }
    data  = {c: {"id": [], "ood": []} for c in conds}
    gw_f  = []
    gw_p  = []
    for run in runs:
        for c in conds:
            data[c]["id"].append(run["conditions"][c]["id_error"])
            data[c]["ood"].append(run["conditions"][c]["ood_error"])
        gw = run["conditions"]["gating_fourier+poly2"]["gate_weights"]
        gw_f.append(gw["fourier"])
        gw_p.append(gw["poly2"])

    n    = len(runs)
    best = min(np.mean(data[c]["ood"]) for c in conds)
    gw_f_m, gw_f_s = agg(gw_f)
    gw_p_m, gw_p_s = agg(gw_p)

    cap = (r"Exp 3.1: Gating network. $f(x) = \sin(x) + 0.3x^2$ (mixed DGP). "
           r"All conditions use MLP. Gate weights are the softmax outputs averaged "
           rf"over the training set. OOD relative error (\%), mean $\pm$ std "
           f"over {n} seeds. Best OOD in bold.")
    out  = table_header(cap, "tab:legacy_old_exp31",
                        "lrrr",
                        r"\textbf{Condition} & \textbf{ID Error (\%)} & "
                        r"\textbf{OOD Error (\%)} & \textbf{Gate weight}")

    for c in conds:
        id_m,  id_s  = agg(data[c]["id"])
        ood_m, ood_s = agg(data[c]["ood"])
        is_best = abs(ood_m - best) < 1e-6
        if c == "fourier":
            gw_str = f"{gw_f_m:.3f} $\\pm$ {gw_f_s:.3f}"
        elif c == "poly2":
            gw_str = f"{gw_p_m:.3f} $\\pm$ {gw_p_s:.3f}"
        else:
            gw_str = "---"
        out += (f"    {lbl_map[c]} & {fmt(id_m, id_s)} & "
                f"{fmt(ood_m, ood_s, bold=is_best)} & {gw_str} \\\\\n")
    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# MAK vector field table
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Exp 2.5 — Fourier manifold alignment
# ---------------------------------------------------------------------------

def table_exp2_5(runs):
    dgps  = ["DGP1_multiplicative", "DGP2_saturated"]
    conds = ["raw_mlp", "fourier_ols", "fourier_mlp", "fno_baseline", "siren_mlp"]
    dgp_tex = {
        "DGP1_multiplicative": r"$G_1$: $\sin(x)\cos(2x)$",
        "DGP2_saturated":      r"$G_2$: $\tanh(5\sin(x)\cos(2x))$",
    }
    data = {d: {c: {"id": [], "ood": []} for c in conds} for d in dgps}
    for run in runs:
        for d in dgps:
            for c in conds:
                data[d][c]["id"].append(run["dgps"][d][c]["id_error"])
                data[d][c]["ood"].append(run["dgps"][d][c]["ood_error"])

    n = len(runs)
    cap = (r"Exp 2.5: Fourier manifold alignment. "
           r"Training $x\in[-\pi,\pi]$, ID $x\in[\pi,2\pi]$, OOD $x\in[2\pi,4\pi]$. "
           r"OOD relative error (\%), mean $\pm$ std over "
           f"{n} seeds. Best OOD per DGP in bold.")
    out = table_header(cap, "tab:exp25", "llrr",
                       r"\textbf{DGP} & \textbf{Condition} & "
                       r"\textbf{ID Error (\%)} & \textbf{OOD Error (\%)}")
    for d in dgps:
        best = min(np.mean(data[d][c]["ood"]) for c in conds)
        first = True
        for c in conds:
            id_m, id_s   = agg(data[d][c]["id"])
            ood_m, ood_s = agg(data[d][c]["ood"])
            is_best = abs(ood_m - best) < 1e-6
            dgp_str = dgp_tex[d] if first else ""
            out += (f"    {dgp_str} & {lbl(c)} & {fmt(id_m, id_s)} & "
                    f"{fmt(ood_m, ood_s, bold=is_best)} \\\\\n")
            first = False
        out += "    \\midrule\n" if d != dgps[-1] else ""
    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# Exp 2.6 — Log-periodic manifold alignment
# ---------------------------------------------------------------------------

def table_exp2_6(runs):
    dgps  = ["DGP1_multiplicative", "DGP2_saturated"]
    conds = ["raw_mlp", "log_ols", "logp_ols", "logp_mlp"]
    dgp_tex = {
        "DGP1_multiplicative": r"$H_1$: $\log y = \sin(\log x)\cos(2\log x)$",
        "DGP2_saturated":      r"$H_2$: $\log y = \tanh(5\sin(\log x)\cos(2\log x))$",
    }
    data = {d: {c: {"id": [], "ood": []} for c in conds} for d in dgps}
    for run in runs:
        for d in dgps:
            for c in conds:
                data[d][c]["id"].append(run["dgps"][d][c]["id_error"])
                data[d][c]["ood"].append(run["dgps"][d][c]["ood_error"])

    n = len(runs)
    cap = (r"Exp 2.6: Log-periodic manifold alignment. "
           r"Training $x\in[e^{-\pi},e^{\pi}]$, ID $x\in[e^{\pi},e^{2\pi}]$, "
           r"OOD $x\in[e^{2\pi},e^{4\pi}]$ (${\approx}12{,}000\times$ scale gap). "
           r"OOD relative error (\%), mean $\pm$ std over "
           f"{n} seeds. Best OOD per DGP in bold.")
    out = table_header(cap, "tab:exp26", "llrr",
                       r"\textbf{DGP} & \textbf{Condition} & "
                       r"\textbf{ID Error (\%)} & \textbf{OOD Error (\%)}")
    for d in dgps:
        best = min(np.mean(data[d][c]["ood"]) for c in conds)
        first = True
        for c in conds:
            id_m, id_s   = agg(data[d][c]["id"])
            ood_m, ood_s = agg(data[d][c]["ood"])
            is_best = abs(ood_m - best) < 1e-6
            dgp_str = dgp_tex[d] if first else ""
            out += (f"    {dgp_str} & {lbl(c)} & {fmt(id_m, id_s)} & "
                    f"{fmt(ood_m, ood_s, bold=is_best)} \\\\\n")
            first = False
        out += "    \\midrule\n" if d != dgps[-1] else ""
    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# Exp 1.3 manifold — combined two-level commitment table
# ---------------------------------------------------------------------------

def table_exp1_3_manifold(runs):
    """Combined Exp 1.3 table (two-level commitment).

    Reads from unified experiment_id "1.3" (all conditions in one JSON).

    G1/G2 layout:
      Block 1 (base):       Fourier × OLS, Fourier × MLP
      thin rule
      Block 2 (foundation): None/Fourier × TabPFN, None/Fourier × TimesFM(cov)

    I1/I2 layout:
      None/transform × SINDy, TabPFN, TimesFM(cov)

    TimesFM rows use the official forecast_with_covariates API when a feature
    map is specified; raw rows use the plain zero-shot forecast.
    """
    import math

    NA = r"\textit{N/A}"

    def _get(dgp_key, cond_key):
        """Return (id_m, id_s, ood_m, ood_s) across seeds, or None if not found."""
        id_vals, ood_vals = [], []
        for r in runs:
            row = r.get("dgps", {}).get(dgp_key)
            if row is None:
                continue
            v = row.get(cond_key)
            if v is None:
                continue
            ie = v.get("id_error")
            oe = float(v.get("ood_error", float("nan")))
            if ie is not None and not math.isnan(float(ie)):
                id_vals.append(float(ie))
            if not math.isnan(oe):
                ood_vals.append(oe)
        if not ood_vals:
            return None
        return (
            float(np.mean(id_vals)) if id_vals else float("nan"),
            float(np.std(id_vals))  if id_vals else float("nan"),
            float(np.mean(ood_vals)),
            float(np.std(ood_vals)),
        )

    # \cmidrule spanning cols 2-5 separates base models from foundation models
    # within each DGP block without repeating the full \midrule weight.
    FOUNDATION_SEP = "    \\cmidrule(lr){2-5}\n"
    SHADE = r"\rowcolor{gray!12}"

    def _fmtrow(dgp_str, fmap_str, model_str, result, best, shaded=False):
        prefix = f"    {SHADE}\n    " if shaded else "    "
        if result is None:
            return f"{prefix}{dgp_str} & {fmap_str} & {model_str} & {NA} & {NA} \\\\\n"
        id_m, id_s, ood_m, ood_s = result
        id_str = "--" if math.isnan(id_m) else fmt(id_m, id_s)
        bold = (not math.isnan(ood_m)) and abs(ood_m - best) < 1e-3
        return (f"{prefix}{dgp_str} & {fmap_str} & {model_str} & "
                f"{id_str} & {fmt(ood_m, ood_s, bold=bold)} \\\\\n")

    n = len(runs)
    cap = (
        r"Exp.~1.3: Two-level commitment. "
        r"Correct feature map (Level~1) is necessary but not sufficient: "
        r"the model class (Level~2) must also match the function's form in "
        r"transformed space. "
        r"\colorbox{gray!12}{\strut Shaded rows} use the correct feature map for the DGP. "
        r"TimesFM~(cov): official \texttt{forecast\_with\_covariates} API "
        r"(transformed features as dynamic numerical covariates). "
        r"OOD relative error (\%), mean $\pm$ std over "
        f"{n} seeds. "
        r"Best OOD per row (finite values only) in bold."
    )
    out = table_header(
        cap, "tab:exp13manifold", "lllrr",
        r"\textbf{DGP} & \textbf{Feature map} & \textbf{Model} & "
        r"\textbf{ID Error (\%)} & \textbf{OOD Error (\%)}"
    )

    # ── Fourier family (G1, G2) ───────────────────────────────────────────────
    # Block 1 (base models):       Fourier × OLS, Fourier × MLP  [shaded]
    # \cmidrule (foundation sep)
    # Block 2 (foundation models): None/Fourier × TabPFN, None/Fourier × TimesFM(cov)
    for dgp_key, dgp_label in [
        ("G1_multiplicative", r"$G_1$: $\sin(x)\cos(2x)$"),
        ("G2_saturated",      r"$G_2$: $\tanh(5\sin(x)\cos(2x))$"),
    ]:
        ood_means = []
        for ck in ("fourier_ols", "fourier_mlp",
                   "tabpfn_raw", "tabpfn_fourier",
                   "timesfm_raw", "timesfm_fourier_cov"):
            r = _get(dgp_key, ck)
            if r and not math.isnan(r[2]):
                ood_means.append(r[2])
        best = min(ood_means) if ood_means else float("nan")

        out += _fmtrow(dgp_label, "Fourier", "OLS",
                       _get(dgp_key, "fourier_ols"), best, shaded=True)
        out += _fmtrow("", "", "MLP",
                       _get(dgp_key, "fourier_mlp"), best, shaded=True)
        out += FOUNDATION_SEP
        out += _fmtrow("", "None",    "TabPFN",
                       _get(dgp_key, "tabpfn_raw"),          best)
        out += _fmtrow("", "Fourier", "TabPFN",
                       _get(dgp_key, "tabpfn_fourier"),      best, shaded=True)
        out += _fmtrow("", "None",    "TimesFM",
                       _get(dgp_key, "timesfm_raw"),         best)
        out += _fmtrow("", "Fourier", "TimesFM (cov)",
                       _get(dgp_key, "timesfm_fourier_cov"), best, shaded=True)
        out += "    \\midrule\n"

    # ── Log-polynomial family (I1, I2) ────────────────────────────────────────
    # None/transform × SINDy, TabPFN, TimesFM(cov)
    logpoly_dgps = [
        ("I1_loglog_arch",
         r"$I_1$: $\log y=2u-u^2,\ u=\log x$",
         "Log-log",
         "raw_sindy", "log_log_sindy", "tabpfn_loglog", "timesfm_loglog_cov"),
        ("I2_logy_sat",
         r"$I_2$: $\log y=1-2e^{-x}$",
         r"Log-$y$",
         "raw_sindy", "log_y_sindy", "tabpfn_logy", "timesfm_logy_cov"),
    ]

    for li, (dgp_key, dgp_label, fmap_label,
             sindy_raw, sindy_tf,
             tabpfn_tf, tfm_tf_cov) in enumerate(logpoly_dgps):

        ood_means = []
        for ck in (sindy_raw, sindy_tf, "tabpfn_raw", tabpfn_tf, "timesfm_raw", tfm_tf_cov):
            r = _get(dgp_key, ck)
            if r and not math.isnan(r[2]):
                ood_means.append(r[2])
        best = min(ood_means) if ood_means else float("nan")

        for dgp_str, fmap_str, model, ck, shaded in [
            (dgp_label, "None",       "SINDy",         sindy_raw,    False),
            ("",        fmap_label,   "SINDy",         sindy_tf,     True),
            ("",        "None",       "TabPFN",        "tabpfn_raw", False),
            ("",        fmap_label,   "TabPFN",        tabpfn_tf,    True),
            ("",        "None",       "TimesFM",       "timesfm_raw",False),
            ("",        fmap_label,   "TimesFM (cov)", tfm_tf_cov,   True),
        ]:
            out += _fmtrow(dgp_str, fmap_str, model, _get(dgp_key, ck), best, shaded=shaded)

        if li < len(logpoly_dgps) - 1:
            out += "    \\midrule\n"

    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def table_summary(all_grouped):
    """Compact overview: one row per (exp, best_condition)."""
    rows = []
    # Each entry: (lookup_eid, display_eid, desc, best_cond, result_type, struct)
    exp_info = [
        ("1.1",       "1.1",  "sin vs. Taylor-9 (MLP)",       "fourier_mlp",          "ood", "conditions"),
        ("1.2",       "1.2",  r"$x^2$ vs. $e^{\alpha x}$ (OLS)", "log_log_ols",        "ood", "dgp_best"),
        ("1.3", "1.3",  "SINDy-OOD diagnostic",          None,                   "rho", "sindy"),
        ("rw1",       "RW1",  "Kepler multivariate (SINDy)",   None,                   "ood", "rw1_best"),
        ("2.1",       "2.1",  "CV vs. near-boundary val.",     None,                   "acc", "selection"),
        ("2.2",       "2.2",  "SHO derivative features (OLS)", "deriv",                "ood", "sho"),
        ("2.3",       "2.3",  "Wavelet features (MLP)",        "wavelet",              "ood", "conditions"),
        ("3.1",       "3.1",  "Gating network (MLP)",          "gating_fourier+poly2", "ood", "conditions"),
    ]

    cap = (r"Summary of key results across all experiments. "
           r"OOD error (\%): mean $\pm$ std over 3 seeds; "
           r"best condition per experiment in bold.")
    out  = table_header(cap, "tab:summary",
                        "llr",
                        r"\textbf{Exp.} & \textbf{Description} & "
                        r"\textbf{Key result}")

    for (eid, disp_eid, desc, best_cond, result_type, struct) in exp_info:
        if eid not in all_grouped:
            continue
        runs = all_grouped[eid]

        if result_type == "ood" and struct == "conditions":
            vals_m = [np.mean([r["conditions"][best_cond]["ood_error"] for r in runs])]
            vals_s = [np.std( [r["conditions"][best_cond]["ood_error"] for r in runs])]
            result_str = f"Best OOD: {vals_m[0]:.1f} $\\pm$ {vals_s[0]:.1f}\\%"

        elif result_type == "ood" and struct == "sho":
            vals_m = [np.mean([r["sho"][best_cond]["ood_error"] for r in runs])]
            vals_s = [np.std( [r["sho"][best_cond]["ood_error"] for r in runs])]
            result_str = f"Best OOD: {vals_m[0]:.1f} $\\pm$ {vals_s[0]:.1f}\\%"

        elif result_type == "ood" and struct == "dgp_best":
            # P1 with log_log (correct), P2 with log_y (correct)
            p1_ood_m = float(np.mean([r["dgps"]["P1_powerlaw"]["log_log_ols"]["ood_error"]
                                      for r in runs]))
            p2_ood_m = float(np.mean([r["dgps"]["P2_exponential"]["log_y_ols"]["ood_error"]
                                      for r in runs]))
            result_str = rf"Correct: $P_1$: {p1_ood_m:.1f}\%, $P_2$: {p2_ood_m:.1f}\%"

        elif result_type == "rho" and struct == "sindy":
            rho  = float(np.mean([r["spearman"]["rho"]   for r in runs]))
            pval = float(np.mean([r["spearman"]["pval"]  for r in runs]))
            n    = int(runs[0]["spearman"]["n_pairs"])
            # Format p-value as a\times10^{b}
            import math as _math
            exp_ = int(_math.floor(_math.log10(abs(pval))))
            man_ = pval / 10**exp_
            pval_str = rf"{man_:.1f}\times10^{{{exp_}}}"
            result_str = (rf"$\rho(\delta_\mathrm{{OOD}},\,\text{{wrong}}) = \mathbf{{{rho:.2f}}}$; "
                          rf"$p={pval_str}$, $n={n}$")

        elif result_type == "ood" and struct == "rw1_best":
            ood_vals = [r["results"].get("log_aM_sindy", {}).get("ood_error", float("nan"))
                        for r in runs]
            ood_vals = [v for v in ood_vals if v == v]  # drop nan
            ood_m = float(np.mean(ood_vals)) if ood_vals else float("nan")
            ood_s = float(np.std(ood_vals))  if ood_vals else float("nan")
            result_str = rf"Best OOD: {ood_m:.1f} $\pm$ {ood_s:.1f}\%"

        elif result_type == "acc" and struct == "selection":
            cv_m  = float(np.mean([r["results"]["cv_accuracy"]  * 100 for r in runs]))
            val_m = float(np.mean([r["results"]["val_accuracy"] * 100 for r in runs]))
            result_str = rf"CV: {cv_m:.0f}\%;\ Val: \textbf{{{val_m:.0f}}}\%"

        else:
            result_str = "---"

        out += f"    {disp_eid} & {desc} & {result_str} \\\\\n"

    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# Hero experiment (Exp RW2): positional encoding as DGP commitment
# ---------------------------------------------------------------------------

PE_ORDER_HERO = ["exact_fourier", "standard_sinusoidal", "rope", "learned_ape", "no_pe"]
PE_DISPLAY_HERO = {
    "exact_fourier":       r"\textbf{Exact Fourier}",
    "standard_sinusoidal": r"Sinusoidal",
    "rope":                r"RoPE",
    "learned_ape":         r"Learned APE",
    "no_pe":               r"No PE (ablation)",
}




def table_hero(hero_runs: list[dict]) -> str:
    """Exp RW2 — Positional encoding as DGP commitment (hero experiment)."""
    from collections import defaultdict
    by_pe: dict[str, list[dict]] = defaultdict(list)
    for r in hero_runs:
        by_pe[r["pe_type"]].append(r)

    chance_err = (1.0 - 1.0 / hero_runs[0]["vocab_size"]) * 100 if hero_runs else 99.61
    n_seeds = max(len(v) for v in by_pe.values()) if by_pe else 1

    cap = (
        r"Exp~RW2: OOD error by positional encoding type on the periodic annotation task. "
        r"Model: 4-layer causal transformer (${\sim}$3M parameters), "
        r"period $P{=}64$, $L_\mathrm{train}{=}256{=}4P$, $L_\mathrm{OOD}{=}128$. "
        r"OOD split: positions $256$--$383$ (never seen during training; full sequence "
        r"fed to model). "
        r"ID error is classification error on in-distribution positions; "
        r"OOD error on positions beyond the training window. "
        rf"Chance OOD error: $1 - 1/256 = {chance_err:.2f}\%$. "
        r"\textbf{Exact Fourier} is the only PE with a theoretical guarantee "
        r"(Lemma~\ref{lem:sin}): $\varphi(L_\mathrm{train}{+}j)=\varphi(j)$, "
        r"so OOD positions map to the same $S^1$ point as in-distribution positions. "
        f"Mean over {n_seeds} seed{'s' if n_seeds > 1 else ''}.  Best OOD in bold."
    )
    col_spec = "lrr"
    header = (r"\textbf{PE type} & "
              r"\textbf{ID Error (\%)} & \textbf{OOD Error (\%)}")
    out = table_header(cap, "tab:hero", col_spec, header)

    best_ood_err = min(
        float(np.mean([r["err_ood"] for r in by_pe[pe]])) * 100
        for pe in PE_ORDER_HERO if pe in by_pe
    ) if by_pe else 0.0

    for pe in PE_ORDER_HERO:
        if pe not in by_pe:
            continue
        runs_pe = by_pe[pe]
        n = len(runs_pe)
        err_id  = [r["err_id"]  * 100 for r in runs_pe]
        err_ood = [r["err_ood"] * 100 for r in runs_pe]
        id_m,  id_s  = agg(err_id)
        ood_m, ood_s = agg(err_ood)
        is_best = abs(ood_m - best_ood_err) < 1e-4
        out += (f"    {PE_DISPLAY_HERO[pe]} & "
                f"{fmt(id_m, id_s, decimals=1)} & "
                f"{fmt(ood_m, ood_s, bold=is_best, decimals=1)} \\\\\n")

    out += "    \\midrule\n"
    out += f"    Chance & --- & {chance_err:.2f} \\\\\n"
    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# Hero-SSM experiment: PE commitment across transformer / Mamba / S4
# ---------------------------------------------------------------------------




def table_exp3_3(runs: list[dict]) -> str:
    """Hero-SSM: ID/OOD error per (model, PE, period). One block per architecture.

    Columns: PE | ID(P=64) OOD(P=64) | ID(P=128) OOD(P=128) | ... | ID(P=512) OOD(P=512)

    Filters to baseline runs only (L_train = 4P, L_ood = 2P, d_state_ssm = 64);
    sweeps A-D are reported in separate tables.
    """
    from collections import defaultdict

    def _is_baseline(r):
        P = r["period"]
        if r.get("L_train") not in (None, 4 * P):
            return False
        if r.get("L_ood") not in (None, 2 * P):
            return False
        # d_state_ssm: keep runs where it's 64 (the original sweep)
        # or missing (very old runs that predate the field).
        ds = r.get("d_state_ssm", 64)
        if ds is not None and ds != 64:
            return False
        return True

    runs = [r for r in runs if _is_baseline(r)]
    if not runs:
        return ""

    # Dedupe: a (model, pe, P, seed) cell may have a pure-baseline file AND a
    # sweep file whose hyperparameters coincide with the baseline. Keep one
    # record per (model, pe, P, seed) so the per-cell mean is over distinct
    # seeds, not over reruns.
    by_seed: dict[tuple, dict] = {}
    for r in runs:
        key = (r["model_type"], r["pe_type"], r["period"], r["seed"])
        by_seed.setdefault(key, r)
    runs = list(by_seed.values())

    periods = sorted({r["period"] for r in runs})
    by_key = defaultdict(list)  # (model, pe, P) -> list[r]
    for r in runs:
        by_key[(r["model_type"], r["pe_type"], r["period"])].append(r)

    chance_err = (1.0 - 1.0 / runs[0]["vocab_size"]) * 100

    n_seeds = max(len(v) for v in by_key.values()) if by_key else 1
    n_total = sum(len(v) for v in by_key.values())

    cap = (
        r"Exp.~3.3 Architectural invariance: ID/OOD error (\%) on the periodic annotation "
        r"task across Transformer, Mamba, S4D and "
        r"$P \in \{" + ", ".join(str(p) for p in periods) + r"\}$ "
        r"($L_\mathrm{train}{=}4P$, $L_\mathrm{OOD}{=}2P$; chance "
        rf"${{\approx}}{chance_err:.2f}\%$). "
        r"\colorbox{gray!12}{\strut Shaded rows} use the correct "
        r"\textbf{Exact Fourier} PE, which yields $\le 0.2\%$ OOD across all "
        r"three backbones (the guarantee belongs to the feature map). "
        rf"Mean over {n_seeds} seeds; {n_total} runs total."
    )
    label = "tab:exp33"

    n_periods = len(periods)
    col_spec  = "l" + "rr" * n_periods  # PE col + (ID, OOD) per period
    head_top  = r"\textbf{PE} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$P{{=}}{p}$}}" for p in periods)
    head_sub  = " & " + " & ".join(
        r"ID & OOD" for _ in periods)
    cmidrule  = " ".join(
        rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(n_periods))

    # Wrap the tabular in \resizebox so the wide 9-column grid fits the
    # text width on \small (8 numeric columns + PE label is too wide otherwise).
    out = (
        f"\\begin{{table}}[ht]\n"
        f"  \\centering\n"
        f"  \\small\n"
        f"  \\caption{{{cap}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\resizebox{{\\linewidth}}{{!}}{{%\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        f"    \\toprule\n"
        f"    {head_top} \\\\\n"
        f"    {cmidrule}\n"
        f"    {head_sub} \\\\\n"
        f"    \\midrule\n"
    )

    for mi, model in enumerate(MODEL_ORDER_ARCHINV):
        # Block header: model name spanning all columns
        out += (f"    \\multicolumn{{{1 + 2 * n_periods}}}{{l}}"
                f"{{\\emph{{{MODEL_DISPLAY_ARCHINV[model]}}}}} \\\\\n")

        # Per-period best OOD across this model's PE rows (for bolding)
        best_per_period = {}
        for P in periods:
            vals = []
            for pe in PE_ORDER_ARCHINV:
                rs = by_key.get((model, pe, P))
                if not rs:
                    continue
                vals.append(float(np.mean([r["err_ood"] for r in rs])) * 100)
            best_per_period[P] = min(vals) if vals else None

        for pe in PE_ORDER_ARCHINV:
            # Skip rope for non-transformer (attention-only)
            if pe == "rope" and model != "transformer":
                continue
            # Skip if no runs for this (model, pe) at all
            if not any(by_key.get((model, pe, P)) for P in periods):
                continue

            cells = []
            for P in periods:
                rs = by_key.get((model, pe, P))
                if not rs:
                    cells.append("---")
                    cells.append("---")
                    continue
                err_id  = [r["err_id"]  * 100 for r in rs]
                err_ood = [r["err_ood"] * 100 for r in rs]
                id_m,  id_s  = agg(err_id)
                ood_m, ood_s = agg(err_ood)
                bold_ood = (best_per_period[P] is not None
                            and abs(ood_m - best_per_period[P]) < 1e-3)
                cells.append(fmt(id_m,  id_s, decimals=1))
                cells.append(fmt(ood_m, ood_s, bold=bold_ood, decimals=1))

            shade = "    \\rowcolor{gray!12}\n" if pe == "exact_fourier" else ""
            out += (f"{shade}    {PE_DISPLAY_ARCHINV[pe]} & "
                    + " & ".join(cells) + " \\\\\n")

        if mi < len(MODEL_ORDER_ARCHINV) - 1:
            out += "    \\midrule\n"

    out += "    \\midrule\n"
    chance_cells = " & ".join(["---", f"{chance_err:.2f}"] * n_periods)
    out += f"    Chance & {chance_cells} \\\\\n"
    out += "    \\bottomrule\n  \\end{tabular}%\n  }\n\\end{table}\n"
    return out














# ---------------------------------------------------------------------------
# RW3: nucleosome positioning (transformer / Mamba / S4 × PE × seed)
# ---------------------------------------------------------------------------

PE_ORDER_RW3 = ["nucleosome_pe", "nucleosome_pe_int10", "nucleosome_pe_int11",
                "codon_pe", "dual_pe", "standard_sinusoidal", "rope",
                "learned_ape", "no_pe"]
PE_DISPLAY_RW3 = {
    "nucleosome_pe":       r"\textbf{Helical} ($P{=}10.4$)",
    "nucleosome_pe_int10": r"Helical ($P{=}10$)",
    "nucleosome_pe_int11": r"Helical ($P{=}11$)",
    "codon_pe":            r"Codon ($P{=}3$, wrong)",
    "dual_pe":             r"Helical $+$ Codon",
    "standard_sinusoidal": r"Sinusoidal",
    "rope":                r"RoPE",
    "learned_ape":         r"Learned APE",
    "no_pe":               r"No PE",
}
MODEL_DISPLAY_RW3 = {"transformer": r"Transformer",
                     "mamba": r"Mamba", "s4": r"S4D"}




def table_exp2_3_rw3_legacy(runs: list[dict]) -> str:
    """RW3: ID + OOD-chrom + OOD-length Pearson r, one block per architecture."""
    from collections import defaultdict

    # Filter out smoke tests (data_mode == 'synthetic') by default; if no real
    # runs exist, fall back to synthetic so the table is still informative.
    real = [r for r in runs if r.get("data_mode") == "real"]
    if real:
        runs = real
        data_label = r"yeast (Brogaard 2012, $\sigma{=}10$ bp smoothing)"
    else:
        data_label = r"\emph{synthetic} ($\sim$10.4 bp helical signal, smoke test)"

    by_key = defaultdict(list)  # (model, pe) -> [r]
    for r in runs:
        by_key[(r["model_type"], r["pe_type"])].append(r)
    if not by_key:
        return ""

    # Splits to report
    splits = [("test",       r"ID test"),
              ("ood_chrom",  r"OOD chr"),
              ("ood_length", r"OOD len")]

    n_seeds = max(len(v) for v in by_key.values())

    cap = (
        rf"Exp.~2.3 (nucleosome positioning, {data_label}): "
        r"Pearson $r$ between predicted and observed nucleosome occupancy. "
        r"\textbf{Helical $P{=}10.4$} commits the model to the biophysical "
        r"DGP (helical pitch of B-form DNA). Codon $P{=}3$ is the wrong "
        r"period and fails even ID. Standard PEs (Sinusoidal, RoPE, Learned "
        r"APE) match ID but degrade on the OOD-length split, where the "
        r"model is evaluated on $W{=}500$ bp windows after training on "
        r"$W{=}200$ bp. "
        rf"Mean over {n_seeds} seed{'s' if n_seeds > 1 else ''}; best per "
        r"architecture/split in bold."
    )

    n_splits = len(splits)
    col_spec = "l" + "r" * n_splits
    head = (r"\textbf{PE} & "
            + " & ".join(rf"\textbf{{{label}}}" for _, label in splits))

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{cap}}}\n  \\label{{tab:rw3}}\n"
           f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
           f"    {head} \\\\\n    \\midrule\n")

    for mi, model in enumerate(["transformer", "mamba", "s4"]):
        rows_present = any((model, pe) in by_key for pe in PE_ORDER_RW3)
        if not rows_present:
            continue
        out += (f"    \\multicolumn{{{1 + n_splits}}}{{l}}"
                f"{{\\emph{{{MODEL_DISPLAY_RW3[model]}}}}} \\\\\n")

        # Best per split for this model
        best_per_split = {}
        for split_key, _ in splits:
            vals = []
            for pe in PE_ORDER_RW3:
                rs = by_key.get((model, pe))
                if not rs:
                    continue
                rs_split = [r["metrics"].get(split_key, {}).get("r")
                            for r in rs]
                rs_split = [v for v in rs_split if v is not None]
                if rs_split:
                    vals.append(float(np.mean(rs_split)))
            best_per_split[split_key] = max(vals) if vals else None

        for pe in PE_ORDER_RW3:
            if pe == "rope" and model != "transformer":
                continue
            rs = by_key.get((model, pe))
            if not rs:
                continue
            cells = []
            for split_key, _ in splits:
                vals = [r["metrics"].get(split_key, {}).get("r") for r in rs]
                vals = [v for v in vals if v is not None]
                if not vals:
                    cells.append("---")
                    continue
                m = float(np.mean(vals))
                s = float(np.std(vals))
                bold = (best_per_split[split_key] is not None
                        and abs(m - best_per_split[split_key]) < 1e-3)
                cell = f"{m:.3f} $\\pm$ {s:.3f}"
                cells.append(f"\\textbf{{{cell}}}" if bold else cell)
            out += f"    {PE_DISPLAY_RW3[pe]} & " + " & ".join(cells) + " \\\\\n"
        if mi < 2:
            out += "    \\midrule\n"
    out += table_footer()
    return out


# ---------------------------------------------------------------------------
# RW3 v2 (CDS-frame): full PE × backbone ablation, learned-frequency Rashomon
# init sweep, window-size sweep on wrong init.
# ---------------------------------------------------------------------------

PE_DISPLAY_CDS = {
    "codon_pe":            r"\textbf{Codon} ($P{=}3$, oracle)",
    "multi_3_resmlp":      r"Codon $\oplus$ $\{10, 40\}$ + zero-init MLP",
    "dual_pe":             r"Codon $\oplus$ Helical (sum)",
    "learned_ape":         r"Learned APE",
    "nucleosome_pe":       r"Helical ($P{=}10.4$, wrong)",
    "standard_sinusoidal": r"Sinusoidal (multi-freq)",
    "no_pe":               r"No PE",
}
PE_ORDER_CDS_TINY = ["codon_pe", "multi_3_resmlp", "dual_pe", "learned_ape",
                     "nucleosome_pe", "standard_sinusoidal", "no_pe"]
PE_ORDER_CDS_FM = ["codon_pe", "no_pe"]
BACKBONE_DISPLAY_CDS = {
    "tiny_transformer": r"\emph{Tiny transformer (random init)}",
    "hyenadna":         r"\emph{HyenaDNA-small (frozen, pretrained)}",
    "caduceus":         r"\emph{Caduceus-PS (frozen, pretrained)}",
}
















# ---------------------------------------------------------------------------
# CDS species-transfer (Tiny + clean labels): the cross-organism cls→biology
# headline. 3 seeds × 4 head variants × 6 organisms.
# ---------------------------------------------------------------------------

# Display order, names and bolding policy.
_CDS_SPECIES_ORDER = ["yeast", "ecoli", "bsubtilis", "spombe",
                       "pfalciparum", "mtuberculosis"]
_CDS_SPECIES_DISPLAY = {
    "yeast":         r"\textit{S.\,cerevisiae}",
    "ecoli":         r"\textit{E.\,coli}",
    "bsubtilis":     r"\textit{B.\,subtilis}",
    "spombe":        r"\textit{S.\,pombe}",
    "pfalciparum":   r"\textit{P.\,falciparum}",
    "mtuberculosis": r"\textit{M.\,tuberculosis}",
}
_CDS_SPECIES_GC = {
    "yeast": "38\\%", "ecoli": "50\\%", "bsubtilis": "44\\%",
    "spombe": "36\\%", "pfalciparum": "19\\%", "mtuberculosis": "65\\%",
}

# (pe, head, period_str)  ->  display row.
# We pair two phase-EQUIVARIANT cls baselines (sinusoidal, learnedAPE) with
# the phase-INV amplitude readout and the biology head. Sinusoidal is the
# fixed-table baseline; learnedAPE is the dominant real-world PE choice
# (BERT/DNABERT default), so showing it collapses too is the strongest
# negative result against standard practice.
_CDS_SPECIES_VARIANTS = [
    ("learned_ape",   "cls",       "[3.0]",        r"learnedAPE + cls"),
    ("sinusoidal",    "cls",       "[3.0]",        r"sinusoidal + cls"),
    ("codon_fourier", "amplitude", "[3.0]",        r"codon\_fourier + amp ($P{=}3$)"),
    ("codon_fourier", "biology",   "[3.0]",        r"\textbf{codon\_fourier + biology ($P{=}3$)}"),
]




def table_exp2_3_species_clean(runs: list[dict]) -> str:
    """Cross-species CDS-vs-nonCDS AUROC under any-strand clean labels.

    Transposed layout: rows = organisms (with GC%), columns = head variants
    + a final Δ-biology-minus-cls column. Single yeast-trained
    TinyTransformer (random init, raw nucleotide tokens, 0.3 M params).
    Evaluated per-organism on full genomes (yeast on held-out chr XI).
    3 seeds, mean ± std.
    """
    if not runs:
        return ""
    by_var = defaultdict(list)
    for r in runs:
        key = (r["pe"], r["head_type"], str(r["periods"]))
        by_var[key].append(r)

    n_seeds = max((len(v) for v in by_var.values()), default=0)
    organisms = _CDS_SPECIES_ORDER

    # Column display names for the variants.
    variant_short = {
        ("sinusoidal",    "cls",       "[3.0]"): r"sin",
        ("learned_ape",   "cls",       "[3.0]"): r"APE",
        ("codon_fourier", "amplitude", "[3.0]"): r"codon",
        ("codon_fourier", "biology",   "[3.0]"): r"codon\,+\,biology",
    }

    cap = (
        r"Cross-species CDS-vs-nonCDS classification, AUROC (mean $\pm$ std "
        rf"over {n_seeds} seeds). Two standard heads (APE and sin) and "
        r"two FE heads (codon, codon+biology). Best variant per organism in bold."
    )

    col_spec = "l r " + "r" * len(_CDS_SPECIES_VARIANTS)
    head_row = (r"\textbf{Organism} & \textbf{GC} & "
                + " & ".join(rf"\textbf{{{variant_short[(pe, head, periods)]}}}"
                             for (pe, head, periods, _label)
                             in _CDS_SPECIES_VARIANTS))

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{cap}}}\n  \\label{{tab:exp23_species_clean}}\n"
           f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
           f"    {head_row} \\\\\n    \\midrule\n")

    mean_table = {}
    std_table = {}
    for (pe, head, periods), seed_rows in by_var.items():
        for org in organisms:
            vals = [r.get(f"eval_{org}_auroc") for r in seed_rows
                    if r.get(f"eval_{org}_auroc") is not None]
            if vals:
                mean_table[(pe, head, periods, org)] = float(np.mean(vals))
                std_table[(pe, head, periods, org)] = float(np.std(vals, ddof=1)) \
                    if len(vals) > 1 else 0.0

    # Best variant per organism (per-row best for the bolding policy).
    best_per_org = {}
    for org in organisms:
        row_means = [mean_table[(pe, head, periods, org)]
                     for (pe, head, periods, _label) in _CDS_SPECIES_VARIANTS
                     if (pe, head, periods, org) in mean_table]
        best_per_org[org] = max(row_means) if row_means else None

    for org in organisms:
        cells = []
        for (pe, head, periods, _label) in _CDS_SPECIES_VARIANTS:
            key = (pe, head, periods, org)
            if key not in mean_table:
                cells.append("---")
                continue
            m = mean_table[key]; s = std_table[key]
            bold = (best_per_org[org] is not None
                    and abs(m - best_per_org[org]) < 1e-3)
            cell = f"{m:.3f}\\,$\\pm$\\,{s:.3f}"
            cells.append(f"\\textbf{{{cell}}}" if bold else cell)
        out += (f"    {_CDS_SPECIES_DISPLAY[org]} & "
                f"{_CDS_SPECIES_GC[org]} & "
                + " & ".join(cells)
                + " \\\\\n")
    out += table_footer()
    return out




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Table builders that do NOT take TFM runs
# ---------------------------------------------------------------------------
# HCS Rashomon scope test (Section sec:hcs_scope in theory.tex)
# ---------------------------------------------------------------------------

def table_hcs_rashomon(runs):
    """HCS scope: OOD MSE per (model, family) plus interpolator-only Rashomon
    spread, mean over seeds. Demonstrates that hierarchical compositional
    structure does NOT collapse the Rashomon set under generic deep-net bias,
    while explicit DGP commitments (block-additive vs. dense polynomial) tell
    the same right-vs-wrong commitment story as Tier 1."""

    families = ["A_hcs", "B_nonhcs", "C_deep_hcs"]
    family_labels = {
        "A_hcs":      r"\textbf{A (HCS shallow)}",
        "B_nonhcs":   r"\textbf{B (non-HCS)}",
        "C_deep_hcs": r"\textbf{C (HCS deep)}",
    }

    interpolators = [
        ("deep_narrow_mlp",       r"deep narrow MLP"),
        ("wide_shallow_mlp",      r"wide shallow MLP"),
        ("deep_narrow_mlp_seed2", r"deep narrow MLP (seed 2)"),
        ("deep_wide_mlp",         r"deep wide MLP"),
        ("krr_rbf",               r"KRR (RBF)"),
        ("krr_poly3",             r"KRR (poly-3)"),
        ("rf",                    r"random forest"),
        ("histgb",                r"histogram GB"),
    ]
    extrapolators = [
        ("block_additive_mlp",  r"block-additive MLP (HCS commitment)"),
        ("dense_poly2_ridge",   r"dense degree-2 polynomial (full mixing)"),
    ]

    # Aggregate ood_mse per (family, model) across seeds; same for interp spread
    agg_data = {f: {n: [] for n, _ in interpolators + extrapolators}
                for f in families}
    interp_spread = {f: [] for f in families}
    for run in runs:
        for f in families:
            fr = run["families"][f]
            for n, _ in interpolators + extrapolators:
                agg_data[f][n].append(fr["per_model"][n]["ood_mse"])
            interp_spread[f].append(fr["interp_spread_normalized"])

    n_seeds = len(runs)
    cfg = runs[0]
    cap = (
        r"Scope test of Section~\ref{sec:hcs_scope}: OOD MSE on standardised "
        r"targets, mean over " + f"{n_seeds}" + r" seeds. "
        r"\emph{Generic interpolators} all reach near-zero ID MSE but disagree "
        r"substantially OOD on every family, including the HCS targets. The "
        r"\emph{HCS-aligned} commitment (block-additive MLP) wins on Family A "
        r"(right commitment), is mediocre on Family B (no architectural "
        r"rejection but no extrapolation gain), and on Family C (commitment-shape "
        r"mismatch). The dense polynomial commitment fails catastrophically on "
        r"all bounded targets. The interpolator-only Rashomon spread (last row) "
        r"is not lower for HCS targets, refuting the conjecture that implicit "
        r"compositional bias collapses $\mathcal{R}(\varepsilon)$. "
        r"Best per family in bold. Setup: "
        rf"$d={cfg['d']}$, $N_{{\text{{train}}}}={cfg['n_train']}$, "
        rf"$N_{{\text{{ood}}}}={cfg['n_ood']}$, ID $\sim \mathcal{{N}}(0, I)$, "
        rf"OOD $\sim \mathcal{{N}}(0, {int(cfg['sigma_out']**2)} I)$, "
        rf"Rashomon threshold ID MSE $\leq {cfg['id_mse_abs_tol']:.2f}$."
    )

    out = table_header(
        cap, "tab:hcs_rashomon", "lrrr",
        r"\textbf{Model} & "
        + " & ".join(family_labels[f] for f in families)
    )

    # Best per family (lowest OOD MSE among interpolators+extrapolators)
    best = {f: min(np.mean(agg_data[f][n])
                   for n, _ in interpolators + extrapolators)
            for f in families}

    def row(name, label):
        cells = []
        for f in families:
            m = float(np.mean(agg_data[f][name]))
            s = float(np.std(agg_data[f][name]))
            is_best = abs(m - best[f]) < 1e-6
            cells.append(fmt(m, s, bold=is_best, decimals=2))
        return f"    {label} & " + " & ".join(cells) + r" \\" + "\n"

    out += r"    \multicolumn{4}{l}{\emph{Generic interpolators}} \\" + "\n"
    for n, lab in interpolators:
        out += row(n, lab)
    out += r"    \midrule" + "\n"
    out += r"    \multicolumn{4}{l}{\emph{Explicit DGP commitments}} \\" + "\n"
    for n, lab in extrapolators:
        out += row(n, lab)
    out += r"    \midrule" + "\n"
    out += (r"    \multicolumn{4}{l}{\emph{Interpolator-only Rashomon spread} "
            r"$/\,\sigma_{\text{OOD}}$} \\" + "\n")
    spread_cells = []
    for f in families:
        m, s = agg(interp_spread[f])
        spread_cells.append(fmt(m, s, bold=False, decimals=2))
    out += (r"    $\overline{\sigma}_{\text{OOD}}^{\text{interp}}\,/\,\sigma_{\text{OOD}}$ & "
            + " & ".join(spread_cells) + r" \\" + "\n")
    out += table_footer()
    return out


def main():
    _here = Path(__file__).resolve().parent   # scripts/
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",
                        default=str(_here.parent / "results"))
    parser.add_argument("--output_dir",
                        default=str(_here.parent / "tables"))
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_grouped = load_results(args.results_dir)

    if not all_grouped:
        print(f"No results found in '{args.results_dir}'.")
        return

    if not args.summary_only:
        # Tables without TFM (mirrors original TABLE_BUILDERS in sorted order).
        for eid, build_fn in sorted(DISPATCH_BY_EID.items()):
            if eid not in all_grouped:
                print(f"Exp {eid}: no results — skipping.")
                continue
            build_fn(all_grouped[eid], args.output_dir)

        # Consolidated SINDy delta_OOD diagnostic table for the appendix.
        toy_runs    = all_grouped.get("1.3")
        exp11_runs  = all_grouped.get("1.1")
        exp12_runs  = all_grouped.get("1.2")
        kepler_runs = all_grouped.get("rw1")
        mak_runs    = all_grouped.get("mak_vf")
        if toy_runs or exp11_runs or exp12_runs or kepler_runs or mak_runs:
            build_sindy_consolidated(
                toy_runs=toy_runs,
                kepler_runs=kepler_runs,
                mak_runs=mak_runs,
                exp11_runs=exp11_runs,
                exp12_runs=exp12_runs,
                output_dir=args.output_dir,
            )

        # Torus benchmark tables (Nagler v2c) — custom filenames.
        for eid, build_fn in DISPATCH_TORUS.items():
            if eid not in all_grouped:
                print(f"  {eid}: no results — skipping.")
                continue
            build_fn(all_grouped[eid], args.output_dir)

        for eid, build_fn in DISPATCH_TORUS_EXTRA:
            if eid not in all_grouped:
                print(f"  {eid}: no results — skipping.")
                continue
            build_fn(all_grouped[eid], args.output_dir)

    # Hero-SSM experiment tables (transformer x mamba x s4 x PE x period).
    hero_ssm_runs = load_exp3_3_results(args.results_dir)
    if hero_ssm_runs:
        for tag, build_fn in DISPATCH_HERO_SSM_SWEEPS:
            build_fn(hero_ssm_runs, args.output_dir)
    else:
        print("  Hero-SSM exp: no results in results/exp3_3/ — skipping.")

    # RW3 v2 (CDS-frame) — three tables.
    cds_runs = load_exp2_3_full_results(args.results_dir)
    if cds_runs:
        build_exp2_3_frame(cds_runs, args.output_dir)
    else:
        print("  CDS-frame ablation: no results in genome_fm_probe_cds_full/ — skipping.")

    cds_sweep = load_exp2_3_learned_sweep(args.results_dir)
    if cds_sweep:
        build_exp2_3_learned_sweep(cds_sweep, args.output_dir)
    else:
        print("  CDS-frame init-period sweep: no results — skipping.")

    cds_fft_summary = load_exp2_3_fft(args.results_dir)
    if cds_fft_summary:
        build_exp2_3_fft(cds_fft_summary, args.output_dir)
    else:
        print("  CDS-frame FFT: no results — skipping.")

    cds_w = load_exp2_3_window_sweep(args.results_dir)
    if cds_w:
        build_exp2_3_window_sweep(cds_w, args.output_dir)
    else:
        print("  CDS-frame window sweep: no results — skipping.")

    # Exp 2.4 — Camelyon17 histopathology (dedicated loader: the result
    # filenames do not match the exp*__*__*.json glob of load_results).
    histo = load_histopath(args.results_dir)
    if histo:
        build_histopath(histo, args.output_dir)
    else:
        print("  Camelyon17 histopathology: no results — skipping.")

    # Compute-resources table for the appendix.
    import resources_tables
    resources_tables.main_inline(results_dir=args.results_dir,
                                 output_dir=args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
