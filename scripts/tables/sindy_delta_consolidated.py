#!/usr/bin/env python3
r"""Single appendix table consolidating SINDy δ_ood diagnostic across all
SINDy-regime experiments where δ is defined.

Sources:
  - Toy battery (\S\ref{sec:exp13}, exp1_4_sindy_coord): 8 univariate DGPs.
  - Exp 1.1 (\S\ref{sec:exp11}, exp1_1_sin): sin(x) and Taylor-9 with
    Fourier vs poly-9 libraries.
  - Exp 1.2 (\S\ref{sec:exp12}, exp1_2_powerlaw_vs_exp): power law and
    exponential with log-log, log-y, raw feature maps (degree-1 poly lib).
  - Kepler third law (\S\ref{sec:rw_kepler}, exprw1_kepler): real
    exoplanet data, 8 multivariate feature maps.
  - MAK vector field (\S\ref{sec:rw_mak}, exp2_4_mak_dynamics): synthetic
    bilinear chemistry ODE, 3 feature maps.

All formulations use the same squared-NMSE normalization; the target
(derivative or static regression) varies by experiment but the diagnostic
intuition is invariant: δ → 0 for canonical-correct φ, δ ≫ 0 otherwise.
Spearman ρ stat reported only when toy battery is present.

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


OUT = Path(__file__).resolve().parents[2] / "tables" / "table_sindy_delta_consolidated.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"


def _load_inputs(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return {
        "toy_runs":    grouped.get("1.3"),
        "kepler_runs": grouped.get("rw1"),
        "mak_runs":    grouped.get("mak_vf"),
        "exp11_runs":  grouped.get("1.1"),
        "exp12_runs":  grouped.get("1.2"),
    }


# === SINDy row helpers (used only by this consolidated table) ===
def _toy_battery_sindy_rows(runs):
    """Generate body rows for the toy SINDy battery (exp1_4_sindy_coord).

    Returns a string of LaTeX `\\\\` rows ready to splice into a consolidated table.
    Mirrors the row structure of `table_exp1_3_sindy` so the appendix table is
    visually consistent.
    """
    import math as _math
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
    }
    for k in list(dgps_agg.keys()):
        if "I1:" in k:
            DGP_LABELS[k] = r"$\log y=2u-u^2$"
        if "I2:" in k:
            DGP_LABELS[k] = r"$\log y=1-2e^{-x}$"

    TF_LABELS = {
        "raw":          r"Raw $x$",
        "fourier":      r"$(\sin x,\cos x)$ (deg.\ 2)",
        "fourier2":     r"$(\sin 2x,\cos 2x)$ (deg.\ 2)",
        "poly2":        r"Poly-2",
        "poly3":        r"Poly-3",
        "fourier-4":    r"Fourier-4 (deg.\ 2)",
        "fourier-6":    r"Fourier-6 (deg.\ 2)",
        "raw x->y":     r"Raw $x{\to}y$",
        "logx,raw y":   r"$\log x$, raw $y$",
        "loglog":       r"Log-log",
        "poly2 x->y":   r"Poly-2 $x{\to}y$",
        "logy,x-feat":  r"$\log y$, feat.\ $x$",
        "logy,v-state": r"$\log y$, state $v$",
        r"logy,e^-x":   r"$\log y$, feat.\ $e^{-x}$",
    }

    def fmt_res(v):
        if _math.isnan(v):
            return r"\textit{---}"
        if v > 9999:
            return r"$\gg 9{,}999$"
        if v >= 100:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"

    dgp_order = [
        "sin(x)", "sin(2x)", "x^2", "x^3+x",
        "G1:sin(x)cos(2x)", "G2:tanh(5sincos)",
    ]
    for k in dgps_agg:
        if "I1:" in k or "I2:" in k:
            dgp_order.append(k)

    n_groups = len([d for d in dgp_order if d in dgps_agg])
    out  = ""
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
            tr_str   = fmt_res(tr_m)
            ood_str  = fmt_res(ood_m)
            if v["correct"]:
                ood_str = f"\\textbf{{{ood_str}}}"
            prefix = "    \\rowcolor{gray!12}\n    " if v["correct"] else "    "
            out += (f"{prefix}{proc_str} & {tf_str} & {k_m:.0f} & "
                    f"{tr_str} & {ood_str} \\\\\n")
        if written < n_groups:
            out += "    \\midrule\n"
    return out


def _kepler_sindy_rows(runs):
    """Generate body rows for Kepler's third law (exprw1_kepler) δ values.

    Returns a string of LaTeX `\\\\` rows ready to splice into a consolidated table.
    Bolds the canonically-correct (log a, log M*) row.
    """
    cond_order = [
        ("log_aM_sindy",     r"$(\log a,\log M_{\mathrm{star}})$",                       True),
        ("log_aMRTe_sindy",  r"$(\log a,\log M_{\mathrm{star}},\log R_{\mathrm{star}},\log T_{\mathrm{eff}})$", False),
        ("log_aTe_sindy",    r"$(\log a,\log T_{\mathrm{eff}})$",                            False),
        ("log_aR_sindy",     r"$(\log a,\log R_{\mathrm{star}})$",                        False),
        ("log_a_sindy",      r"$(\log a)$",                                     False),
        ("log_a_linM_sindy", r"$(\log a,M_{\mathrm{star}})$",                             False),
        ("raw_aM_sindy",     r"$(a,M_{\mathrm{star}})$",                                  False),
        ("lina_logM_sindy",  r"$(a,\log M_{\mathrm{star}})$",                             False),
    ]
    agg_d = {c: {"k": [], "dtr": [], "dood": []} for c, _, _ in cond_order}
    for r in runs:
        for c, _, _ in cond_order:
            v = r["results"].get(c, {})
            agg_d[c]["k"].append(v.get("n_terms", float("nan")))
            agg_d[c]["dtr"].append(v.get("delta_train", float("nan")))
            agg_d[c]["dood"].append(v.get("delta_ood", float("nan")))

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

    n_train = runs[0].get("config", {}).get("n_train", "?")
    n_ood   = runs[0].get("config", {}).get("n_ood",   "?")
    proc_label = (rf"Kepler (real, $n_\mathrm{{train}}={_latex_thousands(n_train)}$, "
                  rf"$n_\mathrm{{OOD}}={_latex_thousands(n_ood)}$)")
    out = ""
    for cond, label, is_canonical in cond_order:
        a = agg_d[cond]
        k_m    = float(np.nanmean(a["k"]))
        dtr_m  = float(np.nanmean(a["dtr"]))
        dood_m = float(np.nanmean(a["dood"]))
        dood_str = fmt_d(dood_m)
        if is_canonical:
            dood_str = f"\\textbf{{{dood_str}}}"
        prefix = "    \\rowcolor{gray!12}\n    " if is_canonical else "    "
        out += (f"{prefix}{proc_label} & {label} & {k_m:.0f} & "
                f"{fmt_d(dtr_m)} & {dood_str} \\\\\n")
    return out


def _exp11_sindy_rows(runs):
    """Generate body rows for exp 1.1 sin(x) vs Taylor-9 SINDy fits.

    Two DGPs (sin x and Taylor-9), two libraries each (Fourier and poly-9).
    Bolds the canonical-correct row for each DGP.
    """
    cond_order_p1 = [
        ("sindy_fourier", r"Fourier $[\sin x,\cos x,1]$",  True),
        ("sindy_raw",     r"Poly-$9$ $[1,x,\dots,x^9]$",   False),
    ]
    cond_order_p2 = [
        ("sindy_p2ctx",         r"Poly-$9$ $[1,x,\dots,x^9]$",  True),
        ("sindy_fourier_p2ctx", r"Fourier $[\sin x,\cos x,1]$", False),
    ]

    def fmt_d(x):
        if not np.isfinite(x):
            return r"\textit{---}"
        if x >= 1e4:
            exp = int(np.floor(np.log10(x)))
            mant = x / 10 ** exp
            return f"${mant:.1f}{{\\times}}10^{{{exp}}}$"
        if x >= 100: return f"{x:.1f}"
        if x >= 1:   return f"{x:.2f}"
        if x >= 0.001: return f"{x:.4f}"
        return f"{x:.2e}"

    def _agg(cond_order):
        agg_d = {c: {"k": [], "dtr": [], "dood": []} for c, _, _ in cond_order}
        for r in runs:
            res = r.get("conditions", r.get("results", {}))
            for c, _, _ in cond_order:
                v = res.get(c, {})
                agg_d[c]["k"].append(v.get("n_terms", float("nan")))
                agg_d[c]["dtr"].append(v.get("delta_train", float("nan")))
                agg_d[c]["dood"].append(v.get("delta_ood", float("nan")))
        return agg_d

    out = ""
    for grp_idx, (cond_order, dgp_label) in enumerate([
        (cond_order_p1, r"$\sin(x)$ \;(Exp.~1.1)"),
        (cond_order_p2, r"Taylor-$9(x)$ \;(Exp.~1.1)"),
    ]):
        agg_d = _agg(cond_order)
        for cond, label, is_canonical in cond_order:
            a = agg_d[cond]
            k_arr = [v for v in a["k"] if v is not None and np.isfinite(v)]
            k_m   = float(np.mean(k_arr)) if k_arr else 0
            dtr_m = float(np.nanmean(a["dtr"]))
            dood_m = float(np.nanmean(a["dood"]))
            dood_str = fmt_d(dood_m)
            if is_canonical:
                dood_str = f"\\textbf{{{dood_str}}}"
            prefix = "    \\rowcolor{gray!12}\n    " if is_canonical else "    "
            out += (f"{prefix}{dgp_label} & {label} & {k_m:.0f} & "
                    f"{fmt_d(dtr_m)} & {dood_str} \\\\\n")
        if grp_idx == 0:
            out += "    \\midrule\n"
    return out


def _exp12_sindy_rows(runs):
    """Generate body rows for exp 1.2 power-law vs exponential SINDy fits.

    Same library (degree-1 polynomial) on three feature maps × two DGPs.
    Bolds the canonical-correct row for each DGP: log-log for the power law,
    log-y for the exponential.
    """
    cond_order = [
        ("sindy_loglog", r"$(\log x) \to \log y$",  "loglog"),
        ("sindy_logy",   r"$x \to \log y$",         "logy"),
        ("sindy_raw",    r"$x \to y$ \;(raw)",      "raw"),
    ]
    canonical = {"P1_powerlaw": "loglog", "P2_exponential": "logy"}
    dgp_label = {
        "P1_powerlaw":    r"$y = x^2$ power law \;(Exp.~1.2)",
        "P2_exponential": r"$y = e^{\alpha(x-1)}$ exp \;(Exp.~1.2)",
    }

    def fmt_d(x):
        if not np.isfinite(x):
            return r"\textit{---}"
        if x >= 1e4:
            exp = int(np.floor(np.log10(x)))
            mant = x / 10 ** exp
            return f"${mant:.1f}{{\\times}}10^{{{exp}}}$"
        if x >= 100: return f"{x:.1f}"
        if x >= 1:   return f"{x:.2f}"
        if x >= 0.001: return f"{x:.4f}"
        return f"{x:.2e}"

    out = ""
    for i_dgp, dgp in enumerate(["P1_powerlaw", "P2_exponential"]):
        agg_d = {c: {"k": [], "dtr": [], "dood": []} for c, _, _ in cond_order}
        for r in runs:
            res = r.get("dgps", {}).get(dgp, {})
            for c, _, _ in cond_order:
                v = res.get(c, {})
                agg_d[c]["k"].append(v.get("n_terms", float("nan")))
                agg_d[c]["dtr"].append(v.get("delta_train", float("nan")))
                agg_d[c]["dood"].append(v.get("delta_ood", float("nan")))
        for cond, label, key in cond_order:
            a = agg_d[cond]
            k_arr = [v for v in a["k"] if v is not None and np.isfinite(v)]
            k_m   = float(np.mean(k_arr)) if k_arr else 0
            dtr_m = float(np.nanmean(a["dtr"]))
            dood_m = float(np.nanmean(a["dood"]))
            is_canonical = (canonical[dgp] == key)
            dood_str = fmt_d(dood_m)
            if is_canonical:
                dood_str = f"\\textbf{{{dood_str}}}"
            prefix = "    \\rowcolor{gray!12}\n    " if is_canonical else "    "
            out += (f"{prefix}{dgp_label[dgp]} & {label} & {k_m:.0f} & "
                    f"{fmt_d(dtr_m)} & {dood_str} \\\\\n")
        if i_dgp == 0:
            out += "    \\midrule\n"
    return out


def _mak_sindy_rows(runs):
    """Generate body rows for the MAK vector-field experiment (exp2_4_mak_dynamics).

    Returns a string of LaTeX `\\\\` rows ready to splice into a consolidated table.
    Bolds the canonically-correct raw $(x_i, x_j)$ row (degree-2 polynomial library
    contains the bilinear $x_i x_j$ term).
    """
    cond_order = [
        ("raw_xixj_sindy",      r"$(x_i,\,x_j)$",            True),
        ("tanh_xixj_sindy",     r"$(\tanh x_i,\,\tanh x_j)$", False),
        ("univariate_xi_sindy", r"$(x_i)$",                  False),
    ]
    agg_d = {c: {"k": [], "dtr": [], "dood": []} for c, _, _ in cond_order}
    for r in runs:
        for c, _, _ in cond_order:
            v = r["conditions"].get(c, {})
            agg_d[c]["k"].append(v.get("n_terms", float("nan")))
            agg_d[c]["dtr"].append(v.get("delta_train", float("nan")))
            agg_d[c]["dood"].append(v.get("delta_ood", float("nan")))

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

    out = ""
    proc_label = r"MAK $\dot x_i = F - B x_i - R x_i x_j$ (synthetic)"
    for cond, label, is_canonical in cond_order:
        a = agg_d[cond]
        k_m    = float(np.nanmean(a["k"]))
        dtr_m  = float(np.nanmean(a["dtr"]))
        dood_m = float(np.nanmean(a["dood"]))
        dood_str = fmt_d(dood_m)
        if is_canonical:
            dood_str = f"\\textbf{{{dood_str}}}"
        prefix = "    \\rowcolor{gray!12}\n    " if is_canonical else "    "
        out += (f"{prefix}{proc_label} & {label} & {k_m:.0f} & "
                f"{fmt_d(dtr_m)} & {dood_str} \\\\\n")
    return out


# %% render — returns tex string (or None when no data)
def render(toy_runs=None, kepler_runs=None, mak_runs=None,
           exp11_runs=None, exp12_runs=None):
    if all(x is None for x in (toy_runs, kepler_runs, mak_runs, exp11_runs, exp12_runs)):
        loaded = _load_inputs()
        toy_runs    = loaded["toy_runs"]
        kepler_runs = loaded["kepler_runs"]
        mak_runs    = loaded["mak_runs"]
        exp11_runs  = loaded["exp11_runs"]
        exp12_runs  = loaded["exp12_runs"]
    if not (toy_runs or kepler_runs or mak_runs or exp11_runs or exp12_runs):
        return None

    sections = []
    if toy_runs:
        sections.append(("toy", _toy_battery_sindy_rows(toy_runs)))
    if exp11_runs:
        sections.append(("exp11", _exp11_sindy_rows(exp11_runs)))
    if exp12_runs:
        sections.append(("exp12", _exp12_sindy_rows(exp12_runs)))
    if kepler_runs:
        sections.append(("kepler", _kepler_sindy_rows(kepler_runs)))
    if mak_runs:
        sections.append(("mak", _mak_sindy_rows(mak_runs)))
    if not sections:
        return None

    rho_str = ""
    if toy_runs:
        rho_vals  = [r["spearman"]["rho"]  for r in toy_runs if "spearman" in r]
        pval_vals = [r["spearman"]["pval"] for r in toy_runs if "spearman" in r]
        n_pairs   = toy_runs[0].get("spearman", {}).get("n_pairs", "?")
        if rho_vals:
            mean_p = float(np.mean(pval_vals))
            # Format scientific notation in proper LaTeX form, e.g. 1.32 \times 10^{-9}.
            if mean_p == 0:
                p_str = r"\le 10^{-308}"
            else:
                exp = int(np.floor(np.log10(abs(mean_p))))
                mant = mean_p / (10 ** exp)
                p_str = rf"{mant:.2f}\times 10^{{{exp}}}"
            rho_str = (rf" Spearman $\rho(\delta_\mathrm{{OOD}},\,\text{{wrong-label}}) "
                       rf"= {float(np.mean(rho_vals)):.2f}$ "
                       rf"($p={p_str}$, $n={n_pairs}$ pairs in toy battery).")

    cap_common_head = (
        r"Consolidated SINDy $\delta_\mathrm{OOD}$ across all SINDy-regime "
        r"experiments \citep{brunton2016sindy}. $\delta_\mathrm{tr}$ and "
        r"$\delta_\mathrm{OOD}$: squared-NMSE "
        r"residual in the SINDy fit space. \colorbox{gray!12}{\strut Shaded rows} "
        r"use a preassigned canonical coordinate, and bolded "
        r"$\delta_\mathrm{OOD} \approx 0$ marks the designated canonical rows. "
        r"$G_2 = \tanh(\cdot)$ has no exact representation in the tested "
        r"libraries (diagnostic silent). The noiseless toy diagnostic varies "
        r"below the printed precision, so repeated seeds give the same displayed "
        r"values. $k$ counts nonzero coefficients."
        rf"{rho_str}"
    )

    # Split into two table environments at the natural boundary between the
    # toy battery (univariate DGPs) and the real / multivariate experiments
    # (Exp 1.1, Exp 1.2, Kepler, MAK), so the table no longer overflows one
    # page. The second environment uses \addtocounter{table}{-1} so both
    # parts share the same number ("Table N (continued)") — references to
    # \ref{tab:sindy_delta_consolidated} keep pointing at the first part.
    toy_keys = {"toy"}
    part_a = [(k, r) for (k, r) in sections if k in toy_keys]
    part_b = [(k, r) for (k, r) in sections if k not in toy_keys]

    def _emit(header, rows_block):
        s = header
        for i, (_, rows) in enumerate(rows_block):
            s += rows
            if i < len(rows_block) - 1:
                s += "    \\midrule\n"
        s += table_footer()
        return s

    if not part_a or not part_b:
        # Fall back to a single table when only one half is present (defensive).
        out = table_header(
            cap_common_head, "tab:sindy_delta_consolidated", "llrrr",
            r"\textbf{DGP / Experiment} & \textbf{Feature map} & "
            r"$\boldsymbol{k}$ & "
            r"$\boldsymbol{\delta_\mathrm{tr}}$ & "
            r"$\boldsymbol{\delta_\mathrm{OOD}}$"
        )
        for i, (_, rows) in enumerate(sections):
            out += rows
            if i < len(sections) - 1:
                out += "    \\midrule\n"
        out += table_footer()
        return out

    cap_a = cap_common_head + r" Toy battery, with additional experiments below."
    cap_b = (
        r"Consolidated SINDy $\delta_\mathrm{OOD}$ (continued): additional "
        r"synthetic and real-data experiments. \colorbox{gray!12}{\strut Shaded "
        r"rows} use a preassigned canonical coordinate, and bold "
        r"$\delta_\mathrm{OOD}\approx 0$ marks the designated canonical rows. "
        r"Residuals use each experiment's fitted target space, and seed counts "
        r"vary by experiment."
    )

    header_row = (
        r"\textbf{DGP / Experiment} & \textbf{Feature map} & "
        r"$\boldsymbol{k}$ & "
        r"$\boldsymbol{\delta_\mathrm{tr}}$ & "
        r"$\boldsymbol{\delta_\mathrm{OOD}}$"
    )

    head_a = table_header(cap_a, "tab:sindy_delta_consolidated", "llrrr", header_row)
    out = _emit(head_a, part_a)

    # Second environment: same table number, no own label, captioned as continued.
    out += "\n\\addtocounter{table}{-1}\n"
    out += (
        f"\\begin{{table}}[ht]\n"
        f"  \\centering\n"
        f"  \\small\n"
        f"  \\caption[Consolidated SINDy $\\delta_\\mathrm{{OOD}}$ diagnostic (continued).]{{{cap_b}}}\n"
        f"  \\begin{{tabular}}{{llrrr}}\n"
        f"    \\toprule\n"
        f"    {header_row} \\\\\n"
        f"    \\midrule\n"
    )
    out = _emit(out, part_b)
    return out


# %% build — CLI / shell entry point
def build(toy_runs=None, kepler_runs=None, mak_runs=None,
          exp11_runs=None, exp12_runs=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = render(toy_runs, kepler_runs, mak_runs, exp11_runs, exp12_runs)
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
