#!/usr/bin/env python3
"""
Visualization for Feature Engineering / DGP Commitment experiments — dispatch shell.

Active result-figure pipeline lives in `figures/results/*.py` (tor_repr, tor_cov,
tor_cap) and is wired through DISPATCH below. The plot_exp1_1 / plot_exp1_2 /
plot_exp2_1 / plot_exp2_2 / plot_exp2_3 / plot_exp2_5 / plot_exp2_6 / plot_exp3_1
/ plot_exp1_3_two_level functions kept later in this file are not in the paper
figure set; if you re-enable any of them, register them in DISPATCH so
sync_paper.py picks up the output.

Usage (from project root):
    conda run -n data_analysis python scripts/plot_results.py
    conda run -n data_analysis python scripts/plot_results.py --exp tor1_representation
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Importing as `python scripts/plot_results.py` puts scripts on sys.path
# so that the figures/ subpackage resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Active-artefact entry points.
from figures.results.exp4_1 import build as build_exp4_1
from figures.results.exp4_2 import build as build_exp4_2
from figures.results.exp4_3 import build as build_exp4_3
from figures.results.exp4_4 import build as build_exp4_4

# Re-export shared style + data helpers so the legacy ad-hoc functions below
# keep working unchanged.
from figures.results._style import (
    RC, C, LABELS, DGP_COLORS, DGP_MARKERS, col, lbl,
)
from figures.results._data import load_results, agg_conditions, hbar


DISPATCH = {
    "4.1": build_exp4_1,
    "4.2":       build_exp4_2,
    "4.3":       build_exp4_3,
    "4.4": build_exp4_4,
}


# ---------------------------------------------------------------------------
# Exp 1.1 — Observational equivalence
# ---------------------------------------------------------------------------

def plot_exp1_1(runs, output_dir):
    agg = agg_conditions(runs)

    # Only plot conditions with both id_error and ood_error; skip NaN (taylor9 sc)
    plot_conds = ["raw_mlp", "fourier_mlp", "fourier_ols"]
    id_m  = [agg[c]["id_error"]["mean"]  for c in plot_conds]
    ood_m = [agg[c]["ood_error"]["mean"] for c in plot_conds]
    id_s  = [agg[c]["id_error"]["std"]   for c in plot_conds]
    ood_s = [agg[c]["ood_error"]["std"]  for c in plot_conds]

    # Use log scale for OOD because of the large range
    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

        # Left: ID vs OOD bar
        hbar(ax1, plot_conds, id_m, ood_m, id_s, ood_s,
             title="In-dist. vs. OOD Error")
        ax1.set_xscale("log")
        ax1.set_xlabel("Relative Error (%, log scale)")

        # Right: show all 4 conditions on OOD only (log scale), incl. taylor9
        all_conds   = ["raw_mlp", "fourier_mlp", "fourier_ols", "taylor9_direct"]
        ood_all     = [agg[c]["ood_error"]["mean"] for c in all_conds]
        ood_all_s   = [agg[c]["ood_error"]["std"]  for c in all_conds]
        colors_all  = [col(c) for c in all_conds]
        labels_all  = [lbl(c) for c in all_conds]
        y = np.arange(len(all_conds))
        for i, (c, color) in enumerate(zip(all_conds, colors_all)):
            ax2.barh(y[i], ood_all[i], xerr=ood_all_s[i], color=color,
                     height=0.5, alpha=0.85, capsize=2,
                     error_kw={"elinewidth": 0.8, "ecolor": "black"})
        ax2.set_yticks(y)
        ax2.set_yticklabels(labels_all)
        ax2.invert_yaxis()
        ax2.set_xscale("log")
        ax2.set_xlabel("OOD Error (%, log scale)")
        ax2.set_title("OOD Error by Feature Type")

        # Note: training-window agreement (max rel. diff. = 1.09%) is described in the caption

        fig.suptitle(r"Exp 1.1: Observational Equivalence: $\sin(x)$ vs. Taylor-9 Poly",
                     fontsize=9, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp1_1_observational_equivalence.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 1.2 — DGP specificity: power law vs exponential
# ---------------------------------------------------------------------------

def plot_exp1_2(runs, output_dir):
    # Aggregate across seeds for each DGP × condition
    dgp_names = ["P1_powerlaw", "P2_exponential"]
    conds     = ["raw_mlp", "log_log_ols", "log_y_ols"]
    dgp_labels = {"P1_powerlaw": r"$P_1$: Power law ($x^2$)",
                  "P2_exponential": r"$P_2$: Exponential ($e^{\alpha(x-1)}$)"}

    # Accumulate per (dgp, cond) across seeds
    data = {d: {c: defaultdict(list) for c in conds} for d in dgp_names}
    for run in runs:
        for dgp in dgp_names:
            for cond in conds:
                for k, v in run["dgps"][dgp][cond].items():
                    data[dgp][cond][k].append(v)

    agg = {d: {c: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                   for k, v in data[d][c].items()}
               for c in conds} for d in dgp_names}

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)

        for ax, dgp in zip(axes, dgp_names):
            ood_m = [agg[dgp][c]["ood_error"]["mean"] for c in conds]
            ood_s = [agg[dgp][c]["ood_error"]["std"]  for c in conds]
            id_m  = [agg[dgp][c]["id_error"]["mean"]  for c in conds]
            id_s  = [agg[dgp][c]["id_error"]["std"]   for c in conds]
            hbar(ax, conds, id_m, ood_m, id_s, ood_s,
                 title=dgp_labels[dgp])
            ax.set_xscale("log")
            ax.set_xlabel("Relative Error (%, log scale)")

        fig.suptitle("Exp 1.2: DGP Specificity: Transformation Matches Its DGP Only",
                     fontsize=9, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp1_2_dgp_specificity.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2.1 — Selection protocol
# ---------------------------------------------------------------------------

def plot_exp2_1(runs, output_dir):
    # Average cv_accuracy and val_accuracy across seeds
    cv_accs  = [r["results"]["cv_accuracy"]  for r in runs]
    val_accs = [r["results"]["val_accuracy"] for r in runs]

    # Selection frequency (counts) from one representative run (seed 42)
    rep = runs[0]
    cv_counts  = rep["results"]["cv_counts"]
    val_counts = rep["results"]["val_counts"]
    transforms = list(cv_counts.keys())
    n_trials   = rep["n_trials"]

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

        # Left: accuracy comparison
        criteria = ["CV (5-fold)", "Near-boundary val."]
        means    = [float(np.mean(cv_accs)) * 100, float(np.mean(val_accs)) * 100]
        stds     = [float(np.std(cv_accs))  * 100, float(np.std(val_accs))  * 100]
        colors   = ["#e74c3c", "#2ecc71"]
        y = np.arange(len(criteria))
        ax1.barh(y, means, xerr=stds, color=colors, height=0.5, alpha=0.85,
                 capsize=3, error_kw={"elinewidth": 1, "ecolor": "black"})
        ax1.axvline(100.0 / rep["config"]["K"], color="grey", ls="--",
                    lw=1, label=f"Random ({100/rep['config']['K']:.0f}%)")
        ax1.set_yticks(y)
        ax1.set_yticklabels(criteria)
        ax1.invert_yaxis()
        ax1.set_xlabel("Selection Accuracy (%)")
        ax1.set_xlim(0, 110)
        ax1.set_title("Accuracy: Identify Correct Transform")
        ax1.legend(fontsize=7, frameon=False)

        # Right: selection frequency distribution (stacked bar)
        t_colors = ["#2ecc71", "#f39c12", "#3498db", "#e74c3c"]
        cv_vals  = [cv_counts[t]  for t in transforms]
        val_vals = [val_counts[t] for t in transforms]
        y2 = np.array([0.3, 0.7])

        left_cv  = 0
        left_val = 0
        for ti, (t, cv_v, val_v, color) in enumerate(zip(transforms, cv_vals, val_vals, t_colors)):
            ax2.barh(y2[0], cv_v,  left=left_cv,  height=0.25, color=color,
                     alpha=0.85, label=lbl(t) if ti == 0 else "_")
            ax2.barh(y2[1], val_v, left=left_val, height=0.25, color=color, alpha=0.85)
            left_cv  += cv_v
            left_val += val_v

        ax2.set_yticks(y2)
        ax2.set_yticklabels(["CV", "Near-bdry. val."])
        ax2.set_xlabel("Selection count (100 trials)")
        ax2.set_title("What Each Criterion Selects")
        ax2.set_xlim(0, n_trials + 5)

        # Color legend for transforms — placed outside axes to avoid overlapping bars
        legend_handles = [mpatches.Patch(color=t_colors[i], label=lbl(transforms[i]))
                          for i in range(len(transforms))]
        ax2.legend(handles=legend_handles, fontsize=7, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        fig.suptitle("Exp 2.1: Near-Boundary Validation vs. Cross-Validation",
                     fontsize=9, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_1_selection_protocol.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2.2 — ODE dynamics
# ---------------------------------------------------------------------------

def plot_exp2_2(runs, output_dir):
    conds = ["raw", "deriv", "poly2"]
    id_m  = [float(np.mean([r["sho"][c]["id_error"]  for r in runs])) for c in conds]
    ood_m = [float(np.mean([r["sho"][c]["ood_error"] for r in runs])) for c in conds]
    id_s  = [float(np.std( [r["sho"][c]["id_error"]  for r in runs])) for c in conds]
    ood_s = [float(np.std( [r["sho"][c]["ood_error"] for r in runs])) for c in conds]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.8, 2.4))
        # Build theory-line handle first, then pass into hbar so there is a single legend
        import matplotlib.lines as mlines
        theory_handle = mlines.Line2D([], [], color="#e74c3c", ls=":", lw=1, alpha=0.7,
                                      label=r"Theory: 50% for 2$\times\,\omega$ shift")
        hbar(ax, conds, id_m, ood_m, id_s, ood_s,
             title=r"$\omega_\mathrm{train}=1.0 \to \omega_\mathrm{OOD}=2.0$",
             extra_handles=[theory_handle])
        ax.set_xscale("log")
        ax.set_xlabel("Relative Error (%, log scale)")
        ax.axvline(50.0, color="#e74c3c", ls=":", lw=1, alpha=0.7)

        fig.suptitle("Exp 2.2: ODE Dynamics: Derivative Features as DGP Commitment",
                     fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_2_ode_dynamics.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2.3 — Wavelet features
# ---------------------------------------------------------------------------

def plot_exp2_3(runs, output_dir):
    agg = agg_conditions(runs)
    conds = ["raw", "fft", "wavelet"]

    id_m  = [agg[c]["id_error"]["mean"]  for c in conds]
    ood_m = [agg[c]["ood_error"]["mean"] for c in conds]
    id_s  = [agg[c]["id_error"]["std"]   for c in conds]
    ood_s = [agg[c]["ood_error"]["std"]  for c in conds]

    cond_labels = {"raw": "Raw signal (MLP)", "fft": "FFT magnitude (MLP)",
                   "wavelet": "Wavelet sub-bands (MLP)"}

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.8, 2.4))
        hbar(ax, conds, id_m, ood_m, id_s, ood_s,
             labels=[cond_labels[c] for c in conds],
             title="Freq. train [1–5 Hz] → OOD [7–12 Hz]")
        ax.set_xlabel("Relative Error (%)")

        fig.suptitle("Exp 2.3: Wavelet Features for Frequency-Shifted Signals",
                     fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_3_wavelet.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 3.1 — Gating network
# ---------------------------------------------------------------------------

def plot_exp3_1(runs, output_dir):
    agg = agg_conditions(runs)
    # For gate_weights, average across seeds
    gw_fourier = float(np.mean([r["conditions"]["gating_fourier+poly2"]["gate_weights"]["fourier"]
                                for r in runs]))
    gw_poly2   = float(np.mean([r["conditions"]["gating_fourier+poly2"]["gate_weights"]["poly2"]
                                for r in runs]))

    conds = ["fourier", "poly2", "raw", "gating_fourier+poly2"]
    ood_m = [agg[c]["ood_error"]["mean"] for c in conds]
    ood_s = [agg[c]["ood_error"]["std"]  for c in conds]
    id_m  = [agg[c]["id_error"]["mean"]  for c in conds]
    id_s  = [agg[c]["id_error"]["std"]   for c in conds]

    cond_labels = {
        "fourier":             "Fourier (MLP)",
        "poly2":               "Poly-2 (MLP)",
        "raw":                 "Raw (MLP)",
        "gating_fourier+poly2":"Gating (MLP)",
    }

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

        # Left: ID vs OOD bars
        hbar(ax1, conds, id_m, ood_m, id_s, ood_s,
             labels=[cond_labels[c] for c in conds],
             title=r"$f(x) = \sin(x) + 0.3x^2$")
        ax1.set_xlabel("Relative Error (%)")

        # Right: gate weights (pie chart)
        pie_vals   = [gw_fourier, gw_poly2]
        pie_labels = [f"Fourier\n{gw_fourier:.2f}", f"Poly-2\n{gw_poly2:.2f}"]
        pie_colors = [C["fourier_mlp"], C["poly2"]]
        wedges, texts = ax2.pie(pie_vals, labels=pie_labels, colors=pie_colors,
                                 startangle=90,
                                 textprops={"fontsize": 8},
                                 wedgeprops={"linewidth": 0.8, "edgecolor": "white"})
        ax2.set_title("Gate weights (DGP posterior)")
        ax2.text(0, -1.45, "≈ equal mixture: periodic + polynomial DGPs",
                 ha="center", fontsize=7, style="italic")

        fig.suptitle("Exp 3.1: Gating Network as DGP Posterior Mixture",
                     fontsize=9, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp3_1_gating.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2.5 — Fourier manifold alignment
# ---------------------------------------------------------------------------

def plot_exp2_5(runs, output_dir):
    dgp_names = ["DGP1_multiplicative", "DGP2_saturated"]
    conds     = ["raw_mlp", "fourier_ols", "fourier_mlp", "fno_baseline", "siren_mlp"]
    dgp_labels = {
        "DGP1_multiplicative": r"$G_1$: $\sin(x)\cos(2x)$",
        "DGP2_saturated":      r"$G_2$: $\tanh(5\sin(x)\cos(2x))$",
    }
    data = {d: {c: defaultdict(list) for c in conds} for d in dgp_names}
    for run in runs:
        for dgp in dgp_names:
            for cond in conds:
                for k, v in run["dgps"][dgp][cond].items():
                    data[dgp][cond][k].append(v)
    agg_d = {d: {c: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                     for k, v in data[d][c].items()}
                 for c in conds} for d in dgp_names}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
        for ax, dgp in zip(axes, dgp_names):
            ood_m = [agg_d[dgp][c]["ood_error"]["mean"] for c in conds]
            ood_s = [agg_d[dgp][c]["ood_error"]["std"]  for c in conds]
            id_m  = [agg_d[dgp][c]["id_error"]["mean"]  for c in conds]
            id_s  = [agg_d[dgp][c]["id_error"]["std"]   for c in conds]
            hbar(ax, conds, id_m, ood_m, id_s, ood_s, title=dgp_labels[dgp])
            ax.set_xscale("log")
            ax.set_xlabel("Relative Error (%, log scale)")
        fig.suptitle("Exp 2.5: Fourier Manifold Alignment", fontsize=9, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_5_manifold_alignment.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 2.6 — Log-periodic manifold alignment
# ---------------------------------------------------------------------------

def plot_exp2_6(runs, output_dir):
    dgp_names = ["DGP1_multiplicative", "DGP2_saturated"]
    conds     = ["raw_mlp", "log_ols", "logp_ols", "logp_mlp"]
    dgp_labels = {
        "DGP1_multiplicative": r"$H_1$: $\log y=\sin(\log x)\cos(2\log x)$",
        "DGP2_saturated":      r"$H_2$: $\log y=\tanh(5\ldots)$",
    }
    data = {d: {c: defaultdict(list) for c in conds} for d in dgp_names}
    for run in runs:
        for dgp in dgp_names:
            for cond in conds:
                for k, v in run["dgps"][dgp][cond].items():
                    data[dgp][cond][k].append(v)
    agg_d = {d: {c: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                     for k, v in data[d][c].items()}
                 for c in conds} for d in dgp_names}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
        for ax, dgp in zip(axes, dgp_names):
            ood_m = [agg_d[dgp][c]["ood_error"]["mean"] for c in conds]
            ood_s = [agg_d[dgp][c]["ood_error"]["std"]  for c in conds]
            id_m  = [agg_d[dgp][c]["id_error"]["mean"]  for c in conds]
            id_s  = [agg_d[dgp][c]["id_error"]["std"]   for c in conds]
            hbar(ax, conds, id_m, ood_m, id_s, ood_s, title=dgp_labels[dgp])
            ax.set_xscale("log")
            ax.set_xlabel("Relative Error (%, log scale)")
        fig.suptitle("Exp 2.6: Log-Periodic Manifold Alignment", fontsize=9, y=1.01)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_6_log_manifold_alignment.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary overview figure
# ---------------------------------------------------------------------------

def plot_summary(all_grouped, output_dir):
    """6-panel overview figure: one panel per experiment tier + SC gap scatter."""
    panels = []
    for eid in ["1.1", "1.2", "2.2", "2.3", "3.1"]:
        if eid in all_grouped:
            panels.append(eid)

    if not panels:
        return

    n = len(panels)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8))
        if n == 1:
            axes = [axes]

        for ax, eid in zip(axes, panels):
            runs = all_grouped[eid]

            if eid == "1.1":
                agg = agg_conditions(runs)
                conds = ["raw_mlp", "fourier_mlp", "fourier_ols"]
                ood_m = [agg[c]["ood_error"]["mean"] for c in conds]
                ood_s = [agg[c]["ood_error"]["std"]  for c in conds]
                y = np.arange(len(conds))
                for i, c in enumerate(conds):
                    ax.barh(y[i], ood_m[i], xerr=ood_s[i], color=col(c),
                            height=0.5, alpha=0.85, capsize=2,
                            error_kw={"elinewidth": 0.8, "ecolor": "black"})
                ax.set_yticks(y); ax.set_yticklabels([lbl(c) for c in conds])
                ax.invert_yaxis(); ax.set_xscale("log")
                ax.set_xlabel("OOD Error (%)"); ax.set_title("1.1: sin(x)")

            elif eid == "1.2":
                dgps = ["P1_powerlaw", "P2_exponential"]
                conds = ["log_log_ols", "log_y_ols"]
                data = {d: {c: defaultdict(list) for c in conds} for d in dgps}
                for run in runs:
                    for dgp in dgps:
                        for cond in conds:
                            data[dgp][cond]["ood_error"].append(
                                run["dgps"][dgp][cond]["ood_error"])
                ood_P1_loglog = float(np.mean(data["P1_powerlaw"]["log_log_ols"]["ood_error"]))
                ood_P1_logy  = float(np.mean(data["P1_powerlaw"]["log_y_ols"]["ood_error"]))
                ood_P2_loglog = float(np.mean(data["P2_exponential"]["log_log_ols"]["ood_error"]))
                ood_P2_logy  = float(np.mean(data["P2_exponential"]["log_y_ols"]["ood_error"]))
                xs = np.array([ood_P1_loglog, ood_P1_logy, ood_P2_loglog, ood_P2_logy])
                lbls_2 = [r"$P_1$+log-log", r"$P_1$+log-$y$",
                          r"$P_2$+log-log", r"$P_2$+log-$y$"]
                colors_2 = [C["log_log_ols"], C["log_y_ols"],
                            C["log_log_ols"], C["log_y_ols"]]
                y = np.arange(4)
                for i in range(4):
                    ax.barh(y[i], xs[i], color=colors_2[i], height=0.5, alpha=0.85)
                ax.set_yticks(y); ax.set_yticklabels(lbls_2)
                ax.invert_yaxis(); ax.set_xscale("log")
                ax.set_xlabel("OOD Error (%)"); ax.set_title("1.2: DGP specificity")

            elif eid in ("2.2", "2.3", "3.1"):
                cond_map = {
                    "2.2": ["raw", "deriv", "poly2"],
                    "2.3": ["raw", "fft", "wavelet"],
                    "3.1": ["fourier", "poly2", "gating_fourier+poly2"],
                }
                title_map = {
                    "2.2": "2.2: ODE / deriv.",
                    "2.3": "2.3: Wavelet",
                    "3.1": "3.1: Gating",
                }
                conds = cond_map[eid]
                if eid == "2.2":
                    ood_m = [float(np.mean([r["sho"][c]["ood_error"] for r in runs])) for c in conds]
                    ood_s = [float(np.std( [r["sho"][c]["ood_error"] for r in runs])) for c in conds]
                else:
                    agg = agg_conditions(runs)
                    ood_m = [agg[c]["ood_error"]["mean"] for c in conds]
                    ood_s = [agg[c]["ood_error"]["std"]  for c in conds]
                y = np.arange(len(conds))
                for i, c in enumerate(conds):
                    ax.barh(y[i], ood_m[i], xerr=ood_s[i], color=col(c),
                            height=0.5, alpha=0.85, capsize=2,
                            error_kw={"elinewidth": 0.8, "ecolor": "black"})
                ax.set_yticks(y); ax.set_yticklabels([lbl(c) for c in conds])
                ax.invert_yaxis()
                ax.set_xlabel("OOD Error (%)")
                ax.set_title(title_map[eid])

        fig.suptitle("OOD Error Across All Experiments (mean ± std, 3 seeds)",
                     fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "summary_overview.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Exp 1.3 — Two-level commitment (combined: Fourier + log-polynomial)
# ---------------------------------------------------------------------------

def plot_exp1_3_two_level(fourier_runs, logpoly_runs, output_dir):
    """Combined figure for Exp 1.3: OOD error (log scale), all 4 DGPs.

    Left panel: Fourier family (G1, G2) — OLS vs MLP
    Right panel: Log-polynomial family (I1, I2) — OLS vs ReLU MLP
    """
    # --- Fourier data ---
    f_dgps  = ["DGP1_multiplicative", "DGP2_saturated"]
    f_lin   = "fourier_ols"
    f_exp   = "fourier_mlp"
    f_labels = {
        "DGP1_multiplicative": r"$G_1$: $\sin(x)\cos(2x)$",
        "DGP2_saturated":      r"$G_2$: $\tanh(5\sin(x)\cos(2x))$",
    }
    f_lin_ood, f_exp_ood = {}, {}
    for dgp in f_dgps:
        f_lin_ood[dgp] = float(np.mean([r["dgps"][dgp][f_lin]["ood_error"] for r in fourier_runs]))
        f_exp_ood[dgp] = float(np.mean([r["dgps"][dgp][f_exp]["ood_error"] for r in fourier_runs]))
        f_lin_ood[dgp + "_std"] = float(np.std([r["dgps"][dgp][f_lin]["ood_error"] for r in fourier_runs]))
        f_exp_ood[dgp + "_std"] = float(np.std([r["dgps"][dgp][f_exp]["ood_error"] for r in fourier_runs]))

    # --- Log-polynomial data ---
    p_dgps = ["I1_loglog_arch", "I2_logy_arch"]
    p_lin  = {"I1_loglog_arch": "log_log_ols",   "I2_logy_arch": "log_y_ols"}
    p_exp  = {"I1_loglog_arch": "log_log_mlp",  "I2_logy_arch": "log_y_mlp"}
    p_labels = {
        "I1_loglog_arch": r"$I_1$: $\log y = u - 2\max(0,u-1),\ u=\log x$",
        "I2_logy_arch":   r"$I_2$: $\log y = x - 2\max(0,x-1)$",
    }
    p_lin_ood, p_exp_ood = {}, {}
    for dgp in p_dgps:
        lc, ec = p_lin[dgp], p_exp[dgp]
        p_lin_ood[dgp] = float(np.mean([r["dgps"][dgp][lc]["ood_error"] for r in logpoly_runs]))
        p_exp_ood[dgp] = float(np.mean([r["dgps"][dgp][ec]["ood_error"] for r in logpoly_runs]))
        p_lin_ood[dgp + "_std"] = float(np.std([r["dgps"][dgp][lc]["ood_error"] for r in logpoly_runs]))
        p_exp_ood[dgp + "_std"] = float(np.std([r["dgps"][dgp][ec]["ood_error"] for r in logpoly_runs]))

    col_lin = "#e74c3c"   # red — wrong (linear) model
    col_exp = "#2ecc71"   # green — correct (expressive) model

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))

        # Left panel: Fourier family
        y_pos = np.array([0, 1])
        h = 0.32
        for i, dgp in enumerate(f_dgps):
            ax1.barh(y_pos[i] + h/2, f_lin_ood[dgp],
                     xerr=f_lin_ood[dgp + "_std"],
                     color=col_lin, height=h, alpha=0.85,
                     capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "black"})
            # poly-2 / MLP bar (may be near 0 — use min clip for log scale)
            exp_val = max(f_exp_ood[dgp], 0.05)
            ax1.barh(y_pos[i] - h/2, exp_val,
                     xerr=f_exp_ood[dgp + "_std"],
                     color=col_exp, height=h, alpha=0.85,
                     capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "black"})
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f_labels[d] for d in f_dgps], fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xscale("log")
        ax1.set_xlabel("OOD Relative Error (%, log scale)")
        ax1.set_title("Fourier family: OLS vs. MLP", fontsize=9)

        # Right panel: Log-polynomial family
        for i, dgp in enumerate(p_dgps):
            ax2.barh(y_pos[i] + h/2, p_lin_ood[dgp],
                     xerr=p_lin_ood[dgp + "_std"],
                     color=col_lin, height=h, alpha=0.85,
                     capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "black"})
            exp_val = max(p_exp_ood[dgp], 0.001)
            ax2.barh(y_pos[i] - h/2, exp_val,
                     xerr=p_exp_ood[dgp + "_std"],
                     color=col_exp, height=h, alpha=0.85,
                     capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "black"})
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([p_labels[d] for d in p_dgps], fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xscale("log")
        ax2.set_xlabel("OOD Relative Error (%, log scale)")
        ax2.set_title(r"Log-polynomial family: OLS vs. ReLU MLP", fontsize=9)

        # Shared legend
        handles = [
            mpatches.Patch(facecolor=col_lin, alpha=0.85, label="Linear (OLS)"),
            mpatches.Patch(facecolor=col_exp, alpha=0.85, label="Expressive (MLP)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.05))

        fig.suptitle(
            "Exp 1.3: Correct feature map alone is not sufficient — model class matters",
            fontsize=9, y=1.02
        )
        fig.tight_layout()
        path = os.path.join(output_dir, "exp1_3_two_level.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing JSON result files (with per-experiment subdirs)")
    parser.add_argument("--output_dir",  default="figures",
                        help="Directory to write figure PDFs")
    parser.add_argument("--exp", default=None,
                        help="Plot a single experiment (e.g. 1.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_grouped = load_results(args.results_dir)

    if not all_grouped:
        print(f"No results found in '{args.results_dir}'.")
        return

    if args.exp:
        if args.exp not in all_grouped:
            print(f"No results for {args.exp}.")
            return
        if args.exp in DISPATCH:
            DISPATCH[args.exp](all_grouped[args.exp], args.output_dir)
        else:
            print(f"No plot function for {args.exp}.")
        return

    for eid, runs in sorted(all_grouped.items()):
        if eid in DISPATCH:
            print(f"Plotting Exp {eid} ({len(runs)} seed(s))...")
            DISPATCH[eid](runs, args.output_dir)
        else:
            print(f"Exp {eid}: no plot function — skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
