#!/usr/bin/env python3
"""
Concept / schematic figures: dispatch shell.

Active concept figures live in `figures/concept/*.py` and are wired through
DISPATCH below. The legacy plot_exp1_2_concept / plot_exp_rw1_concept /
plot_exp2_1_concept / plot_exp3_1_concept / plot_exp2_5_concept /
plot_exp2_6_concept / plot_hero_ssm / plot_cds_species_concept functions are
kept later in this file for ad-hoc / appendix-supplement use; if you re-enable
any of them, register it in DISPATCH so sync_paper.py picks up the output.

Usage (from project root):
    conda run -n data_analysis python scripts/plot_concept.py
    conda run -n data_analysis python scripts/plot_concept.py --exp 1.1
    conda run -n data_analysis python scripts/plot_concept.py --output_dir figures
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Importing as `python scripts/plot_concept.py` puts scripts on sys.path
# so that the figures/ subpackage resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Active-artefact entry points.
from figures.concept.exp1_1 import build as build_exp1_1
from figures.concept.exp1_3 import build as build_exp1_3
from figures.concept.exp3_3_archinv import build as build_exp3_3
from figures.concept.exp2_3_cds import build as build_exp2_3_cds
from figures.concept.exp2_3_fft import build as build_exp2_3_fft
from figures.concept.expfm_v_histopath import build as build_histopath
from figures.concept.expfm_v_histopath import build_panel as build_histopath_panel

# Re-export shared style + helpers so the legacy ad-hoc functions below (and
# external callers like main_plot_concept.py) keep working.
from figures.concept._style import (
    RC,
    C_TRAIN, C_NEAR, C_OOD,
    B_TRAIN, B_NEAR, B_OOD,
    C_SIN, C_POLY, C_EXP,
)
from figures.concept._helpers import shade_zone, zone_legend_handles, taylor9


DISPATCH = {
    "1.1":     build_exp1_1,
    "1.3":     build_exp1_3,
    "hero":    build_exp3_3,
    "cds":     build_exp2_3_cds,
    "cds_fft": build_exp2_3_fft,
    "histopath": build_histopath,
    "histopath_panel": build_histopath_panel,
}


# ── Exp 1.2 concept ──────────────────────────────────────────────────────────

def plot_exp1_2_concept(output_dir: str):
    """
    Three panels, one per feature space.  Each panel shows P1=x^2 and
    P2=exp(alpha*(x-1)) across training [1,2], near-boundary [2,3], OOD [2,10],
    in the coordinate system of the feature space.

    Panel 1 – raw (x, y)         : P1 is concave-up, P2 grows faster OOD
    Panel 2 – log-log (log x, log y) : P1 is linear (slope 2), P2 is curved
    Panel 3 – log-y  (x, log y)  : P2 is linear, P1 is curved
    """
    alpha = np.log(4)

    x_all  = np.linspace(1.0, 10.0, 800)
    y_P1   = x_all ** 2
    y_P2   = np.exp(alpha * (x_all - 1))

    # Zone boundaries (training [1,2], OOD [2,10])
    x_train_end = 2.0
    x_ood_end   = 10.0

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

        panel_specs = [
            # (title, xdata, ydata_P1, ydata_P2, xlabel, ylabel)
            ("Raw space",
             x_all, y_P1, y_P2, r"$x$", r"$y$"),
            (r"Log-log space  $(\log x,\, \log y)$",
             np.log(x_all), np.log(y_P1), np.log(y_P2),
             r"$\log x$", r"$\log y$"),
            (r"Log-$y$ space  $(x,\, \log y)$",
             x_all, np.log(y_P1), np.log(y_P2),
             r"$x$", r"$\log y$"),
        ]

        # Zone boundary in each coordinate system
        def zone_x(panel_idx, xval):
            if panel_idx == 1:
                return np.log(xval)
            return xval

        for pi, (ax, (title, xd, y1d, y2d, xl, yl)) in \
                enumerate(zip(axes, panel_specs)):

            z1 = zone_x(pi, x_train_end)
            z2 = zone_x(pi, x_ood_end)
            x0 = xd[0]

            ax.axvspan(x0, z1, alpha=0.18, color=C_TRAIN, zorder=0)
            ax.axvspan(z1, z2, alpha=0.18, color=C_OOD,   zorder=0)
            ax.axvline(z1, color=B_TRAIN, lw=0.7, ls=":", zorder=1)

            ax.plot(xd, y1d, color=C_SIN,  lw=2,        label=r"$P_1 = x^2$", zorder=3)
            ax.plot(xd, y2d, color=C_EXP,  lw=2, ls="--",
                    label=r"$P_2 = e^{\alpha(x-1)}$", zorder=3)

            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(title)

            ax.autoscale(axis='y')
            ytop = ax.get_ylim()[1]

            mid1 = (x0 + z1) / 2
            mid2 = (z1 + z2) / 2

            for mx, txt, col in [
                (mid1, "train", B_TRAIN),
                (mid2, "OOD",   B_OOD),
            ]:
                ax.text(mx, ytop * 0.97, txt, ha="center", va="top",
                        fontsize=6.5, color=col, fontweight="bold")

            # Annotate which DGP is "straight" in this space
            if pi == 1:
                ax.text(0.04, 0.25, r"$P_1$: linear (slope 2)",
                        transform=ax.transAxes, fontsize=6.5, color=C_SIN)
                ax.text(0.04, 0.13, r"$P_2$: curved",
                        transform=ax.transAxes, fontsize=6.5, color=C_EXP)
            elif pi == 2:
                ax.text(0.04, 0.94, r"$P_2$: linear (slope $\alpha$)",
                        transform=ax.transAxes, fontsize=6.5, color=C_EXP,
                        va="top")
                ax.text(0.04, 0.80, r"$P_1$: curved",
                        transform=ax.transAxes, fontsize=6.5, color=C_SIN,
                        va="top")

        # Shared legend (bottom): DGPs + zones (train, OOD only)
        dgp_handles = [
            mlines.Line2D([], [], color=C_SIN, lw=2, label=r"$P_1 = x^2$"),
            mlines.Line2D([], [], color=C_EXP, lw=2, ls="--",
                          label=r"$P_2 = e^{\alpha(x-1)}$"),
            mpatches.Patch(fc=C_TRAIN, ec=B_TRAIN, lw=0.8, label="Training"),
            mpatches.Patch(fc=C_OOD,   ec=B_OOD,   lw=0.8, label="OOD"),
        ]
        fig.legend(handles=dgp_handles, fontsize=7, frameon=False,
                   loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))

        fig.suptitle(
            r"Exp 1.2: Power law vs. exponential: same training endpoints, "
            r"divergent OOD, in three feature spaces",
            fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp1_2_concept.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ── Exp RW1 concept ──────────────────────────────────────────────────────────

def plot_exp_rw1_concept(output_dir: str):
    """
    Three panels: multivariate Kepler feature-engineering concept.

    Panel 1 – log-log scatter (log a, log T) colored by M_star tercile:
               univariate fit has scatter correlated with stellar mass.
    Panel 2 – Corrected coordinate (1.5 log a - 0.5 log M): scatter collapses to a line.
    Panel 3 – OOD error bar chart for key feature-map / model-class conditions.
    """
    from datasets import load_dataset

    ds = load_dataset("juliensimon/nasa-exoplanets", split="train")
    df = ds.to_pandas()
    sub = df[["pl_orbsmax", "pl_orbper", "st_mass"]].dropna()
    sub = sub[(sub["pl_orbsmax"] > 0) & (sub["pl_orbper"] > 0) & (sub["st_mass"] > 0)]
    a_all = sub["pl_orbsmax"].values
    T_all = sub["pl_orbper"].values / 365.25
    M_all = sub["st_mass"].values

    A_SPLIT = 0.5
    A_VIS   = 20.0

    mask_tr = a_all < A_SPLIT
    mask_od = (a_all >= A_SPLIT) & (a_all < A_VIS)

    a_tr = a_all[mask_tr];  T_tr = T_all[mask_tr];  M_tr = M_all[mask_tr]
    a_od = a_all[mask_od];  T_od = T_all[mask_od];  M_od = M_all[mask_od]

    la_tr = np.log(a_tr);  lT_tr = np.log(T_tr);  lM_tr = np.log(M_tr)
    la_od = np.log(a_od);  lT_od = np.log(T_od);  lM_od = np.log(M_od)

    # Univariate log-log OLS (log a -> log T)
    A1 = np.column_stack([la_tr, np.ones_like(la_tr)])
    w1, _, _, _ = np.linalg.lstsq(A1, lT_tr, rcond=None)

    # Bivariate OLS (log a, log M -> log T)
    A2 = np.column_stack([la_tr, lM_tr, np.ones_like(la_tr)])
    w2, _, _, _ = np.linalg.lstsq(A2, lT_tr, rcond=None)

    # M_star tercile coloring (training + OOD together for color scale)
    lM_all_valid = np.concatenate([lM_tr, lM_od])
    q33, q67 = np.percentile(lM_all_valid, [33, 67])
    def m_color(lM):
        c = np.where(lM < q33, 0, np.where(lM < q67, 1, 2))
        return c
    cmap3 = ["#4e8ec4", "#7bc67e", "#e07b55"]   # low / mid / high M_star
    labels3 = [r"low $M_{\mathrm{star}}$", r"mid $M_{\mathrm{star}}$", r"high $M_{\mathrm{star}}$"]

    rng = np.random.default_rng(42)
    def pick(n, *arrays):
        idx = rng.choice(len(arrays[0]), min(n, len(arrays[0])), replace=False)
        return tuple(a[idx] for a in arrays)

    la_trs, lT_trs, lM_trs = pick(500, la_tr, lT_tr, lM_tr)
    la_ods, lT_ods, lM_ods = pick(200, la_od, lT_od, lM_od)

    # Corrected coordinate: z = 1.5*log a - 0.5*log M
    z_tr = 1.5 * la_trs - 0.5 * lM_trs
    z_od = 1.5 * la_ods - 0.5 * lM_ods
    # Intercept from bivariate OLS
    intercept = float(w2[2])

    # OOD error bar data (mean over 3 seeds, hardcoded from experimental results)
    bar_labels = [
        r"$\mathrm{lin}\,a$, $\log M_{\mathrm{star}}$ (OLS)",
        r"$\mathrm{lin}\,a$, $\log M_{\mathrm{star}}$ (MLP)",
        r"$\log a$ only (OLS)",
        r"$\log a$, $\log R_{\mathrm{star}}$ (OLS)",
        r"$\log a$, $\log M_{\mathrm{star}}$ (MLP)",
        r"$\log a$, $\log M_{\mathrm{star}}$ (TabPFN)",
        r"$\log a$, $\log M_{\mathrm{star}}$ (OLS)",
        r"$\log a$, $\log M_{\mathrm{star}}$ (SINDy)",
    ]
    bar_vals = [9999.0, 82.5, 21.2, 16.3, 67.2, 24.5, 3.8, 2.1]
    bar_colors = [
        "#e74c3c",   # wrong transform OLS -- red
        "#c0392b",   # wrong transform MLP -- dark red
        "#c09a30",   # log a only -- amber
        "#c09a30",   # proxy R
        "#9b59b6",   # MLP correct -- purple
        "#8e44ad",   # TabPFN correct
        "#2ecc71",   # OLS correct -- green
        "#27ae60",   # SINDy correct -- dark green
    ]
    bar_clip = 500.0
    bar_vals_clipped = [min(v, bar_clip) for v in bar_vals]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

        # ── Panel 1: Log-log scatter, colored by M_star tercile ──────────────
        ax = axes[0]
        la_split = np.log(A_SPLIT)
        la_vis   = np.log(A_VIS)
        la_min   = la_trs.min() - 0.1

        ax.axvspan(la_min,   la_split, alpha=0.18, color=C_TRAIN, zorder=0)
        ax.axvspan(la_split, la_vis,   alpha=0.18, color=C_OOD,   zorder=0)
        ax.axvline(la_split, color=B_TRAIN, lw=0.7, ls=":", zorder=1)

        for gi in [0, 1, 2]:
            m_tr = m_color(lM_trs) == gi
            m_od = m_color(lM_ods) == gi
            ax.scatter(la_trs[m_tr], lT_trs[m_tr], s=4, alpha=0.45,
                       color=cmap3[gi], zorder=3, label=labels3[gi])
            ax.scatter(la_ods[m_od], lT_ods[m_od], s=4, alpha=0.35,
                       color=cmap3[gi], zorder=3, marker="^")

        # Univariate OLS line
        la_grid = np.linspace(la_min, la_vis, 300)
        ax.plot(la_grid, w1[0] * la_grid + w1[1], color="black",
                lw=1.8, zorder=4, label=rf"OLS slope$={w1[0]:.2f}$")
        # Residuals are M_star-correlated => annotation
        ax.text(0.04, 0.97,
                r"Residuals $\propto -\frac{1}{2}\log M_{\mathrm{star}}$",
                transform=ax.transAxes, fontsize=6.5, va="top", color="#555")

        ax.set_xlabel(r"$\log a$")
        ax.set_ylabel(r"$\log T$")
        ax.set_title(r"Univariate: $\log a \to \log T$" "\n"
                     r"(colored by $M_{\mathrm{star}}$ tercile)")
        ax.legend(fontsize=6.0, frameon=False, loc="lower right", ncol=1)
        for xm, txt, col in [(la_split - 1.0, "train", B_TRAIN),
                              (la_split + 1.0, "OOD",   B_OOD)]:
            ax.text(xm, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 4.0,
                    txt, ha="center", fontsize=7, color=col, fontweight="bold")

        # ── Panel 2: Corrected coordinate collapses scatter ───────────────────
        ax = axes[1]
        z_min = min(z_tr.min(), z_od.min()) - 0.2
        z_max = max(z_tr.max(), z_od.max()) + 0.2
        # Shade by origin (train / OOD)
        ax.scatter(z_tr, lT_trs, s=4, alpha=0.45, color=B_TRAIN,
                   zorder=3, label="Train")
        ax.scatter(z_od, lT_ods, s=4, alpha=0.35, color=B_OOD,
                   zorder=3, marker="^", label="OOD")
        # Reference line: log T = z + intercept
        z_grid = np.linspace(z_min, z_max, 300)
        ax.plot(z_grid, z_grid + intercept, color=C_SIN, lw=2.0, zorder=4,
                label=r"$\log T = z + c$, slope$=1$")
        ax.plot(z_grid, z_grid + intercept, color="grey", lw=1.0, ls=":", zorder=3)
        ax.text(0.04, 0.97,
                r"$z = \frac{3}{2}\log a - \frac{1}{2}\log M_{\mathrm{star}}$",
                transform=ax.transAxes, fontsize=7.5, va="top", color=C_SIN)
        ax.text(0.04, 0.86,
                "Scatter collapses; train\n and OOD on same line",
                transform=ax.transAxes, fontsize=6.5, va="top", color="#555")
        ax.set_xlabel(r"$z = \frac{3}{2}\log a - \frac{1}{2}\log M_{\mathrm{star}}$")
        ax.set_ylabel(r"$\log T$")
        ax.set_title("Correct coordinate\n"
                     r"($\log a + \log M_{\mathrm{star}}$ bivariate OLS)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")

        # ── Panel 3: OOD error bar chart ──────────────────────────────────────
        ax = axes[2]
        y_pos = np.arange(len(bar_labels))
        bars = ax.barh(y_pos, bar_vals_clipped, color=bar_colors, alpha=0.85,
                       height=0.65)
        # Mark clipped bars with ">500"
        for i, (v, vc) in enumerate(zip(bar_vals, bar_vals_clipped)):
            if v > bar_clip:
                ax.text(vc + 8, y_pos[i], r"$\gg$9999%", va="center",
                        fontsize=6.5, color="#e74c3c")
            else:
                ax.text(vc + 8, y_pos[i], f"{v:.1f}%", va="center", fontsize=6.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=6.5)
        ax.set_xlabel("OOD relative error (%)")
        ax.set_title("Feature map $\\times$ model class\n(OOD error, lower is better)")
        ax.set_xlim(0, bar_clip * 1.3)
        ax.axvline(21.2, color=C_POLY, lw=0.8, ls="--", zorder=0,
                   label=r"$\log a$ only")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.grid(axis="y", alpha=0.0)
        ax.spines["left"].set_visible(False)

        fig.tight_layout()
        path = os.path.join(output_dir, "exp_rw1_concept.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ── Exp 2.1 concept ──────────────────────────────────────────────────────────

def plot_exp2_1_concept(output_dir: str):
    """
    Two panels:
      Left  – sin(x) with three zones + two hypotheses (fourier, poly9).
              The two are indistinguishable in training, begin to separate in
              near-boundary, and completely diverge in OOD.
      Right – Error profile of fourier_ols vs poly9_ols across the three zones,
              showing the information gap that near-boundary validation exploits.
    """
    x_full = np.linspace(-np.pi - 0.1, 3*np.pi + 0.1, 1200)
    y_sin  = np.sin(x_full)
    y_t9   = np.clip(taylor9(x_full), -5, 5)

    # Residuals in each zone (absolute |prediction - true|)
    def zone_residuals(y_true, y_pred, x, x0, x1):
        mask = (x >= x0) & (x <= x1)
        return np.abs(y_true[mask] - y_pred[mask]).mean() * 100  # rough %

    x_pts = x_full
    res_fourier_train = 0.3   # read from table
    res_fourier_near  = 0.12
    res_fourier_ood   = 0.3
    res_poly9_train   = 0.1
    res_poly9_near    = 827.0
    res_poly9_ood     = 16799.0

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

        # ── Left: prediction curves ───────────────────────────────────────
        ax = ax1
        shade_zone(ax, -np.pi, 0,       C_TRAIN, B_TRAIN, "Training",  zorder=0)
        shade_zone(ax,  0,     np.pi,   C_NEAR,  B_NEAR,  "Near-bdry", zorder=0)
        shade_zone(ax,  np.pi, 3*np.pi, C_OOD,   B_OOD,   "OOD",       zorder=0)

        ax.plot(x_full, y_sin, color=C_SIN,  lw=2.2, label=r"$\sin(x)$ (true)",  zorder=4)
        ax.plot(x_full, y_t9,  color=C_POLY, lw=2.0, ls="--",
                label="Taylor-9 / poly commit.", zorder=3)

        ax.set_xlim(-np.pi - 0.1, 3*np.pi + 0.1)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$f(x)$")
        ax.set_xticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"])

        # Zone labels
        for xm, txt, col in [
            (-np.pi/2, "train",     B_TRAIN),
            ( np.pi/2, "near-bdry", B_NEAR),
            ( 2*np.pi, "OOD",       B_OOD),
        ]:
            ax.text(xm, 4.1, txt, ha="center", fontsize=7.5,
                    color=col, fontweight="bold")

        # Callout: indistinguishable in training
        ax.annotate("indistinguishable\nin training",
                    xy=(-np.pi/3, np.sin(-np.pi/3)),
                    xytext=(-2.1, -3.0), fontsize=6.5, ha="center", color="grey",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_title("Predictions across zones")

        # ── Right: error profile per zone (log scale bar chart) ──────────
        ax = ax2
        zones    = ["Training", "Near-bdry\nval.", "OOD eval."]
        x_pos    = np.arange(3)
        w        = 0.32

        f_errs = [res_fourier_train, res_fourier_near,  res_fourier_ood]
        p_errs = [res_poly9_train,   res_poly9_near,    res_poly9_ood]

        bars_f = ax.bar(x_pos - w/2, f_errs, w, color=C_SIN,  alpha=0.85,
                        label=r"Fourier commit. (correct)", zorder=3)
        bars_p = ax.bar(x_pos + w/2, p_errs, w, color=C_POLY, alpha=0.85,
                        label="Poly commit. (wrong)", zorder=3)

        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(zones)
        ax.set_ylabel("Mean error (%, log scale)")
        ax.set_title("Error profile across zones")

        # Zone background colors
        for xi, fc, ec in zip(x_pos, [C_TRAIN, C_NEAR, C_OOD],
                                     [B_TRAIN, B_NEAR, B_OOD]):
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.12, color=fc, zorder=0)

        # Annotate CV-visible zone
        ax.text(0, ax.get_ylim()[0] * 1.5,
                "CV sees\nonly this →", ha="center", fontsize=6.5,
                color=B_TRAIN, style="italic")

        ax.legend(fontsize=7, frameon=False, loc="upper left")

        fig.suptitle(
            r"Exp 2.1: Near-boundary validation vs. CV: "
            r"the zones that each criterion can see",
            fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_1_concept.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ── Exp 3.1 concept ──────────────────────────────────────────────────────────

def plot_exp3_1_concept(output_dir: str):
    """
    Two panels for Exp 3.1 (gating network, mixed DGP f(x) = sin(x) + 0.3x²).

    Training [-π, 0.8π], near-boundary [0.8π, π], OOD [π, 3π].

    Panel 1 – target and components across three zones.
    Panel 2 – schematic error profile: Fourier head vs. Poly-2 head vs. Gating.
    """
    x_full = np.linspace(-np.pi - 0.1, 3*np.pi + 0.1, 1200)

    def f_target(x): return np.sin(x) + 0.3 * x**2
    def f_sin(x):    return np.sin(x)
    def f_poly(x):   return 0.3 * x**2

    X_TRAIN_END = 0.8 * np.pi
    X_NEAR_END  = np.pi
    X_OOD_END   = 3 * np.pi

    C_TARGET = "#2ecc71"    # green
    C_SIN_C  = "#3498db"    # blue  – sin component
    C_POLY_C = "#f39c12"    # orange – poly component
    C_GATE   = "#8e44ad"    # purple – gating

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))

        # ── Panel 1: target and components ───────────────────────────────
        ax = ax1
        shade_zone(ax, -np.pi,      X_TRAIN_END, C_TRAIN, B_TRAIN, "Training", zorder=0)
        shade_zone(ax, X_TRAIN_END, X_NEAR_END,  C_NEAR,  B_NEAR,  "Near-bdry", zorder=0)
        shade_zone(ax, X_NEAR_END,  X_OOD_END,   C_OOD,   B_OOD,   "OOD", zorder=0)

        ax.plot(x_full, f_target(x_full), color=C_TARGET, lw=2.2,
                label=r"$f(x)=\sin(x)+0.3x^2$", zorder=4)
        ax.plot(x_full, f_sin(x_full),    color=C_SIN_C,  lw=1.4, ls="--",
                label=r"$\sin(x)$ component", zorder=3)
        ax.plot(x_full, f_poly(x_full),   color=C_POLY_C, lw=1.4, ls=":",
                label=r"$0.3x^2$ component", zorder=3)

        ax.set_xlim(-np.pi - 0.1, X_OOD_END + 0.1)
        ax.set_ylim(-4, 30)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$f(x)$")
        ax.set_xticks([-np.pi, 0, 0.8*np.pi, np.pi, 2*np.pi, 3*np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$0.8\pi$", r"$\pi$",
                             r"$2\pi$", r"$3\pi$"])

        for xm, txt, col in [
            ((-np.pi + X_TRAIN_END) / 2, "train",     B_TRAIN),
            ((X_TRAIN_END + X_NEAR_END) / 2, "near-bdry", B_NEAR),
            ((X_NEAR_END + X_OOD_END) / 2,   "OOD",       B_OOD),
        ]:
            ax.text(xm, 28, txt, ha="center", fontsize=7.5,
                    color=col, fontweight="bold")

        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_title(r"Mixed DGP: $\sin(x)$ and $0.3x^2$ components")

        # ── Panel 2: schematic error profile ─────────────────────────────
        ax = ax2
        zones    = ["Training", "Near-boundary\nval.", "OOD"]
        x_pos    = np.arange(3)
        w        = 0.25

        # Illustrative (schematic) error values
        # Fourier head: low ID, moderate OOD (misses polynomial trend)
        # Poly-2 head:  low ID, high OOD (misses periodicity at large x)
        # Gating:       lowest across all zones
        errs = {
            "fourier": [3.5,  6.0,  14.0],
            "poly2":   [4.0, 10.0,  22.0],
            "gating":  [2.0,  3.5,   9.0],
        }
        colors_b = [C_SIN_C, C_POLY_C, C_GATE]
        labels_b = ["Fourier head", "Poly-2 head", "Gating (near-bdry. loss)"]

        for i, (key, color, label) in enumerate(
                zip(["fourier", "poly2", "gating"], colors_b, labels_b)):
            ax.bar(x_pos + (i - 1) * w, errs[key], w, color=color,
                   alpha=0.85, label=label, zorder=3)

        # Zone background tints
        for xi, fc, ec in zip(x_pos, [C_TRAIN, C_NEAR, C_OOD],
                                      [B_TRAIN, B_NEAR, B_OOD]):
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.10, color=fc, zorder=0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(zones)
        ax.set_ylabel("Relative error (%, schematic)")
        ax.set_title("Error profile: single heads vs. gating")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.text(0.98, 0.96,
                "Near-boundary loss drives gate\nto $\\approx$0.5/0.5 mixture",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
                color="grey", style="italic")

        fig.suptitle(
            r"Exp 3.1: Gating network as DGP posterior: "
            r"$f(x)=\sin(x)+0.3x^2$ across three zones",
            fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp3_1_concept.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ── Exp 2.5 concept ──────────────────────────────────────────────────────────

def plot_exp2_5_concept(output_dir: str):
    """
    Three panels for Exp 2.5 (Fourier manifold alignment, periodic functions).

    Panel 1 (left):  DGP f(x) = sin(x)*cos(2x) in raw x-space with three zones.
    Panel 2 (center): Fourier feature space S^1 circle showing training tiles
                       the full circle and OOD maps to the same circle.
    Panel 3 (right):  2D scatter of (sin(x), sin(2x)) for training (blue) and
                       OOD (red) confirming they overlap on the same compact region.
    """
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

        # ── Panel 1: DGP in raw x-space ──────────────────────────────────────
        ax = axes[0]
        x_full = np.linspace(-np.pi - 0.1, 4 * np.pi + 0.1, 1200)
        y_dgp  = np.sin(x_full) * np.cos(2 * x_full)

        shade_zone(ax, -np.pi,    np.pi,     C_TRAIN, B_TRAIN, "Training")
        shade_zone(ax,  np.pi,  2*np.pi,     C_NEAR,  B_NEAR,  "ID")
        shade_zone(ax, 2*np.pi, 4*np.pi,     C_OOD,   B_OOD,   "OOD")

        ax.plot(x_full, y_dgp, color=C_SIN, lw=2.0,
                label=r"$G_1(x)=\sin(x)\cos(2x)$", zorder=3)

        ax.set_xlim(-np.pi - 0.1, 4*np.pi + 0.1)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$G_1(x)$")
        ax.set_xticks([-np.pi, 0, np.pi, 2*np.pi, 4*np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$4\pi$"])

        for xm, txt, col in [
            (0,           "train",  B_TRAIN),
            (1.5*np.pi,   "ID",     B_NEAR),
            (3.0*np.pi,   "OOD",    B_OOD),
        ]:
            ax.text(xm, 1.18, txt, ha="center", fontsize=7.5,
                    color=col, fontweight="bold")

        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.set_title(r"DGP $G_1(x)$ across zones")

        # ── Panel 2: Fourier feature space S^1 ───────────────────────────────
        ax = axes[1]
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.8, 1.7)
        ax.axis("off")
        ax.set_title(r"Fourier manifold $\mathcal{Z} = S^1$")

        # Full circle (background)
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), color="lightgrey", lw=1, zorder=1)

        # Training x in [-pi, pi] -> full S^1
        x_tr = np.linspace(-np.pi, np.pi, 600)
        ax.plot(np.sin(x_tr), np.cos(x_tr), color=B_TRAIN, lw=5,
                alpha=0.7, solid_capstyle="round", zorder=3,
                label=r"Train $[-\pi,\pi]$")

        # OOD x in [2pi, 4pi] -> same full circle (dashed outer ring)
        x_od = np.linspace(2*np.pi, 4*np.pi, 600)
        ax.plot(np.sin(x_od) * 1.13, np.cos(x_od) * 1.13,
                color=B_OOD, lw=1.8, ls="--", alpha=0.6, zorder=2,
                label=r"OOD $[2\pi,4\pi]$")

        ax.text(0.0, -1.55,
                "Training tiles $S^1$; OOD maps to the same circle.\n"
                "OOD = interpolation in feature space.",
                ha="center", fontsize=6.5, color="grey", style="italic")

        ax.legend(fontsize=6.5, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.60, 1.05))

        # ── Panel 3: 2D scatter (sin(x), sin(2x)) ───────────────────────────
        ax = axes[2]
        rng = np.random.default_rng(42)

        x_train_pts = rng.uniform(-np.pi, np.pi, 400)
        x_ood_pts   = rng.uniform(2*np.pi, 4*np.pi, 400)

        ax.scatter(np.sin(x_train_pts), np.sin(2 * x_train_pts),
                   s=6, alpha=0.5, color=B_TRAIN, label=r"Train $[-\pi,\pi]$", zorder=3)
        ax.scatter(np.sin(x_ood_pts),   np.sin(2 * x_ood_pts),
                   s=6, alpha=0.4, color=B_OOD, marker="x",
                   label=r"OOD $[2\pi,4\pi]$", zorder=2)

        ax.set_xlabel(r"$\sin(x)$")
        ax.set_ylabel(r"$\sin(2x)$")
        ax.set_title(r"Scatter $(\sin x,\,\sin 2x)$: train $\approx$ OOD")

        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.text(0.02, 0.97,
                "Both tile the same bounded region.\nOOD is not extrapolation.",
                transform=ax.transAxes, fontsize=6.5, va="top",
                color="grey", style="italic")

        fig.suptitle(
            "Exp 2.5: Fourier features place training and OOD on the same compact manifold",
            fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_5_concept.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")
        plt.close(fig)


# ── Exp 2.6 concept ──────────────────────────────────────────────────────────

def plot_exp2_6_concept(output_dir: str):
    """
    Three panels for Exp 2.6 (log-periodic manifold alignment).

    Panel 1 (left):  log y target in raw x-space (log scale). Training and OOD
                     regions with 12000x scale gap.
    Panel 2 (center): Same data in log(x) space. Training u in [-pi,pi],
                       OOD u in [2pi,4pi] -- still disjoint.
    Panel 3 (right):  Log-periodic feature scatter (sin(log x), cos(log x))
                       for training (blue) and OOD (red) -- both tile unit circle.
    """
    X_LO_TRAIN = np.e ** (-np.pi)
    X_HI_TRAIN = np.e **  np.pi
    X_LO_OOD   = np.e ** (2 * np.pi)
    X_HI_OOD   = np.e ** (4 * np.pi)

    def dgp1(x):
        u = np.log(np.clip(x, 1e-12, None))
        return np.sin(u) * np.cos(2 * u)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

        # ── Panel 1: raw x-space (log scale x-axis) ─────────────────────────
        ax = axes[0]
        # Dense grid covering training and OOD
        x_tr_dense  = np.exp(np.linspace(-np.pi, np.pi, 400))
        x_ood_dense = np.exp(np.linspace(2*np.pi, 4*np.pi, 400))

        ax.semilogx(x_tr_dense,  dgp1(x_tr_dense),  color=B_TRAIN, lw=1.8,
                    label=r"Train $[e^{-\pi},e^{\pi}]$", zorder=3)
        ax.semilogx(x_ood_dense, dgp1(x_ood_dense), color=B_OOD, lw=1.8, ls="--",
                    label=r"OOD $[e^{2\pi},e^{4\pi}]$", zorder=3)

        ax.axvspan(X_LO_TRAIN, X_HI_TRAIN, alpha=0.15, color=C_TRAIN, zorder=0)
        ax.axvspan(X_LO_OOD,   X_HI_OOD,   alpha=0.15, color=C_OOD,   zorder=0)

        ax.set_xlabel(r"$x$ (log scale)")
        ax.set_ylabel(r"$\log y = G_1(\log x)$")
        ax.set_title(r"DGP in raw $x$-space (${\approx}12{,}000\times$ gap)")
        ax.legend(fontsize=7, frameon=False, loc="lower right")

        ax.text(0.02, 0.97,
                r"$12{,}000\times$ scale gap",
                transform=ax.transAxes, fontsize=7, va="top",
                color=B_OOD, fontweight="bold")

        # ── Panel 2: log x space ─────────────────────────────────────────────
        ax = axes[1]
        u_tr  = np.linspace(-np.pi, np.pi, 400)
        u_ood = np.linspace(2*np.pi, 4*np.pi, 400)

        ax.axvspan(-np.pi, np.pi,    alpha=0.15, color=C_TRAIN, zorder=0)
        ax.axvspan(2*np.pi, 4*np.pi, alpha=0.15, color=C_OOD,   zorder=0)
        ax.axvline(np.pi,   color=B_TRAIN, lw=0.7, ls=":", zorder=1)
        ax.axvline(2*np.pi, color=B_OOD,   lw=0.7, ls=":", zorder=1)

        ax.plot(u_tr,  np.sin(u_tr)  * np.cos(2 * u_tr),  color=B_TRAIN, lw=1.8,
                label=r"Train $u\in[-\pi,\pi]$", zorder=3)
        ax.plot(u_ood, np.sin(u_ood) * np.cos(2 * u_ood), color=B_OOD, lw=1.8, ls="--",
                label=r"OOD $u\in[2\pi,4\pi]$", zorder=3)

        ax.set_xlabel(r"$u = \log x$")
        ax.set_ylabel(r"$\log y$")
        ax.set_title(r"In $\log x$ space: regions still disjoint")
        ax.legend(fontsize=7, frameon=False, loc="lower right")

        ax.text(0.02, 0.97,
                "Plain log does not help:\ntrain and OOD are still disjoint.",
                transform=ax.transAxes, fontsize=6.5, va="top",
                color="grey", style="italic")

        # ── Panel 3: log-periodic feature scatter ───────────────────────────
        ax = axes[2]
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.8, 1.7)
        ax.axis("off")
        ax.set_title(r"Log-periodic manifold: $(\sin(\log x),\cos(\log x))$")

        # Full circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), color="lightgrey", lw=1, zorder=1)

        # Training in log-periodic space
        u_tr_pts = np.linspace(-np.pi, np.pi, 600)
        ax.plot(np.sin(u_tr_pts), np.cos(u_tr_pts), color=B_TRAIN, lw=5,
                alpha=0.7, solid_capstyle="round", zorder=3,
                label=r"Train $[e^{-\pi},e^{\pi}]$")

        # OOD in log-periodic space (outer dashed ring)
        u_ood_pts = np.linspace(2*np.pi, 4*np.pi, 600)
        ax.plot(np.sin(u_ood_pts) * 1.13, np.cos(u_ood_pts) * 1.13,
                color=B_OOD, lw=1.8, ls="--", alpha=0.6, zorder=2,
                label=r"OOD $[e^{2\pi},e^{4\pi}]$")

        ax.text(0.0, -1.55,
                "Log-periodic features are bounded:\n"
                "OOD = interpolation in feature space.",
                ha="center", fontsize=6.5, color="grey", style="italic")

        ax.legend(fontsize=6.5, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.60, 1.05))

        fig.suptitle(
            r"Exp 2.6: Log-periodic features collapse $12{,}000\times$ scale gap onto a compact manifold",
            fontsize=9, y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "exp2_6_concept.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")
        plt.close(fig)


def _load_exp3_3_results(results_root: Path) -> list | None:
    """Load hero_ssm result JSONs, recovering pe_type for SSMs."""
    import json
    d = results_root / "hero_ssm"
    if not d.is_dir():
        return None
    out = []
    for p in sorted(d.glob("exp3_3__*.json")):
        with open(p) as f:
            r = json.load(f)
        # Filename: exp3_3__<model>__[pe]__P<P>__seed<S>__L<L>__O<LO>__ds<ds>__<ts>.json
        # When pe_type == "no_pe", non-transformer files drop the pe segment.
        parts = p.stem.split("__")
        r["model_type"] = parts[1]
        r["pe_type"]    = "no_pe" if parts[2].startswith("P") else parts[2]
        out.append(r)
    return out or None


def plot_hero_ssm(output_dir: str):
    """Hero-SSM headline result: OOD error per (architecture, PE, period)
    at the baseline configuration ($L_{\\mathrm{train}}=4P$, $L_{\\mathrm{OOD}}=2P$,
    $d_{\\mathrm{state}}=64$). Three panels (one per backbone), heat-coded.
    Cells with no data are rendered as hatched grey (e.g. mamba runs still
    queued)."""
    import json
    results_root = Path(__file__).resolve().parent.parent / "results"
    runs = _load_exp3_3_results(results_root)
    if not runs:
        print("  hero_ssm: no results in results/hero_ssm — skipping.")
        return

    # Filter to baseline configuration (matches table_exp3_3)
    def _is_baseline(r):
        P = r["period"]
        if r.get("L_train") not in (None, 4 * P): return False
        if r.get("L_ood")   not in (None, 2 * P): return False
        ds = r.get("d_state_ssm", 64)
        if ds is not None and ds != 64: return False
        return True
    runs = [r for r in runs if _is_baseline(r)]
    if not runs:
        print("  hero_ssm: no baseline runs after filter — skipping.")
        return

    MODELS = [("transformer", "Transformer"),
              ("mamba",       "Mamba"),
              ("s4",          "S4D")]
    # PE order: best (exact) → worst.  Skip rope for SSM panels (attention-only).
    PES = [
        ("exact_fourier",         "Exact Fourier",        True),
        ("learned_fourier_kfreq", "Learned Fourier (k=4)", True),
        ("learned_fourier_1freq", "Learned Fourier (k=1)", True),
        ("learnable_sinusoidal",  "Learnable Sinusoidal",  True),
        ("standard_sinusoidal",   "Sinusoidal (fixed)",    True),
        ("rope",                  "RoPE",                  False),
        ("learned_ape",           "Learned APE",           True),
        ("no_pe",                 "No PE",                 True),
    ]
    periods = sorted({r["period"] for r in runs})

    # Aggregate OOD error % per (model, pe, P)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in runs:
        agg[(r["model_type"], r["pe_type"], r["period"])].append(r["err_ood"] * 100)

    chance_pct = (1.0 - 1.0 / runs[0].get("vocab_size", 256)) * 100

    cmap = plt.get_cmap("RdYlGn_r")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=chance_pct)

    with plt.rc_context({**RC, "axes.grid": False}):
        # Extra width on the left for PE row labels.
        fig = plt.figure(figsize=(13.0, 0.42 * len(PES) + 2.0))
        gs = fig.add_gridspec(1, len(MODELS) + 1,
                              width_ratios=[1.0, 1.0, 1.0, 0.05],
                              left=0.18, right=0.95, top=0.86, bottom=0.16,
                              wspace=0.10)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(MODELS))]
        cbar_ax = fig.add_subplot(gs[0, -1])

        for col, (ax, (model, mlabel)) in enumerate(zip(axes, MODELS)):
            # Filter PE rows that apply to this model (drop rope for non-transformer)
            rows = [(pe, lbl) for pe, lbl, on_ssm in PES
                    if model == "transformer" or on_ssm]
            data = np.full((len(rows), len(periods)), np.nan)
            for i, (pe, _) in enumerate(rows):
                for j, P in enumerate(periods):
                    vals = agg.get((model, pe, P))
                    if vals:
                        data[i, j] = float(np.mean(vals))

            im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")

            # Hatch missing cells so "leave space for mamba" reads visually.
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isnan(data[i, j]):
                        ax.add_patch(mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=True, facecolor="#eeeeee",
                            edgecolor="#bbbbbb", lw=0.4,
                            hatch="///", zorder=2))
                        ax.text(j, i, "—", ha="center", va="center",
                                fontsize=7, color="#888888", zorder=3)
                    else:
                        v = data[i, j]
                        # Pick text colour for legibility against the heat
                        tc = "white" if (v < 15 or v > 80) else "#222222"
                        txt = "0" if v < 0.05 else (f"{v:.0f}" if v >= 5 else f"{v:.1f}")
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=7, color=tc, zorder=3, fontweight="bold"
                                if v < 1 else "normal")

            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels([f"$P{{=}}{p}$" for p in periods], fontsize=7)
            if col == 0:
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels([lbl for _, lbl in rows], fontsize=7)
            else:
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels([])
            ax.set_xlim(-0.5, len(periods) - 0.5)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            ax.set_title(mlabel, fontsize=9, fontweight="bold")
            ax.tick_params(axis="both", which="both", length=0)

        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label("OOD error (%)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Hatched-grey legend patch for missing cells
        miss = mpatches.Patch(facecolor="#eeeeee", edgecolor="#bbbbbb",
                              hatch="///", label="not yet run")
        axes[0].legend(handles=[miss], loc="upper left",
                       bbox_to_anchor=(0.0, -0.08), frameon=False, fontsize=7)

        fig.suptitle(
            "Exp RW2 Arch-Inv: OOD error (%) at baseline "
            r"($L_\mathrm{train}{=}4P$, $L_\mathrm{OOD}{=}2P$, "
            r"$d_\mathrm{state}{=}64$). "
            f"Chance: {chance_pct:.1f}%.",
            fontsize=9, y=0.97)
        path = os.path.join(output_dir, "archinv.pdf")
        fig.savefig(path)
        print(f"  Saved -> {path}")
        plt.close(fig)


def _load_exp2_3_species_clean_summary(results_root: Path):
    """Aggregate Phase C clean-label species sweep results.

    Reads `exp2_3_species_clean_full/seed*/summary.json`.
    Returns a dict keyed by (variant_label, organism) -> (mean, std)
    over seeds, plus the ordered organism list.
    """
    rows = []
    for subdir in ("exp2_3_species_clean_full",
                   "exp2_3_species_clean_ape"):
        d = results_root / subdir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("seed*/summary.json")):
            with open(p) as f:
                data = json.load(f)
            for r in data.get("rows", []):
                rows.append(r)
    if not rows:
        return None, None
    # Variants of interest: cls baseline, amp(P=3), sorted(P=3), biology(P=3).
    variants = [
        ("sinusoidal",    "cls",       "[3.0]", "cls_sin"),
        ("learned_ape",   "cls",       "[3.0]", "cls_ape"),
        ("codon_fourier", "amplitude", "[3.0]", "amp"),
        ("codon_fourier", "biology",   "[3.0]", "biology"),
    ]
    organisms = ["yeast", "ecoli", "bsubtilis", "spombe",
                 "pfalciparum", "mtuberculosis"]
    out = {}
    for pe, head, periods, label in variants:
        for org in organisms:
            vals = []
            for r in rows:
                if (r.get("pe") == pe and r.get("head_type") == head
                        and str(r.get("periods")) == periods):
                    v = r.get(f"eval_{org}_auroc")
                    if v is not None:
                        vals.append(v)
            if vals:
                out[(label, org)] = (float(np.mean(vals)), float(np.std(vals, ddof=1))
                                     if len(vals) > 1 else 0.0)
    return out, organisms


def plot_cds_species_concept(output_dir: str):
    """Cross-species CDS-vs-nonCDS concept figure (Phase C, clean labels).

    Three panels:
      1. DGP schematic: a CDS read in 5'->3' orientation, with the three
         biology hypotheses highlighted (codon period 3, no in-frame
         stops, biological AA composition).
      2. Per-organism AUROC bars: 4 head variants × 6 organisms. Mtb cls
         falls below the 0.5 chance line.
      3. Catastrophic gap: Δ biology − cls per species, with the synthetic
         hero gap (0.66) drawn as a dashed reference line.
    """
    results_root = Path(__file__).resolve().parent.parent / "results"
    summary, organisms = _load_exp2_3_species_clean_summary(results_root)
    if summary is None:
        print("  [cds_species] no Phase C results, skipping")
        return

    # Display labels (italic, abbreviated)
    org_display = {
        "yeast":         r"$\mathit{S.\,cer.}$",
        "ecoli":         r"$\mathit{E.\,coli}$",
        "bsubtilis":     r"$\mathit{B.\,sub.}$",
        "spombe":        r"$\mathit{S.\,pom.}$",
        "pfalciparum":   r"$\mathit{P.\,fal.}$",
        "mtuberculosis": r"$\mathit{M.\,tub.}$",
    }
    org_gc = {"yeast": 38, "ecoli": 50, "bsubtilis": 44, "spombe": 36,
              "pfalciparum": 19, "mtuberculosis": 65}
    head_order  = ["cls_ape", "cls_sin", "amp", "biology"]
    head_display = {"cls_sin": "sin\\,+\\,cls", "cls_ape": "APE\\,+\\,cls",
                     "amp": "cf\\,+\\,amp", "biology": "cf\\,+\\,biology"}
    head_colour = {"cls_sin": B_OOD, "cls_ape": "#c87137",
                    "amp": "#888888", "biology": C_SIN}

    import matplotlib.pyplot as plt
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.5, 3.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.55, 1.05], wspace=0.32)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        # ── Panel 1: CDS read schematic with three biology hypotheses ────
        ax = ax0
        ax.axis("off")
        seq = "ATGGCATTAGAACCAGTC"   # 18 bp = 6 codons, no in-frame stop
        n = len(seq)
        CW = 1.0
        Y_SEQ = 2.05
        CH = 0.55
        # Box per nucleotide
        for i, base in enumerate(seq):
            r = mpatches.FancyBboxPatch(
                (i + 0.06, Y_SEQ - CH/2), CW - 0.12, CH,
                boxstyle="round,pad=0.02",
                fc="#e8edf2", ec="#888", lw=0.4, zorder=2)
            ax.add_patch(r)
            ax.text(i + CW/2, Y_SEQ, base, ha="center", va="center",
                    fontsize=8.5, fontfamily="monospace", color="#222",
                    fontweight="bold", zorder=3)
        # Codon brackets
        for k in range(n // 3):
            xL = k * 3 + 0.06
            xR = (k + 1) * 3 - 0.06
            ax.annotate("", xy=(xR, Y_SEQ + CH/2 + 0.10),
                        xytext=(xL, Y_SEQ + CH/2 + 0.10),
                        arrowprops=dict(arrowstyle="|-|",
                                        color=B_TRAIN, lw=0.7,
                                        mutation_scale=2.5))
        ax.text(n/2, Y_SEQ + CH/2 + 0.32, "codon period $P{=}3$",
                ha="center", va="bottom", fontsize=7.5, color=B_TRAIN,
                fontstyle="italic")
        # Biology hypothesis annotations (stacked below)
        bullets = [
            (r"$\mathbf{H_1}$: codon period $P{=}3$  $\to$  $|\mathrm{FFT}|@P{=}3$",
             1.05),
            (r"$\mathbf{H_2}$: zero stops in true frame  $\to$  count(\texttt{TAA},\texttt{TAG},\texttt{TGA})",
             0.45),
            (r"$\mathbf{H_3}$: protein-typical AA  $\to$  per-frame 21-class histogram",
             -0.15),
        ]
        for txt, y in bullets:
            ax.text(0.0, y, txt, ha="left", va="center",
                    fontsize=8, color="#333")
        ax.text(-0.5, Y_SEQ, r"$x_t$:", ha="right", va="center",
                fontsize=8, color="#666")
        ax.set_xlim(-2.5, n + 0.5)
        ax.set_ylim(-0.6, 3.1)
        ax.set_title("CDS = 3 universal-genetic-code hypotheses",
                     fontsize=9, pad=8)

        # ── Panel 2: per-organism AUROC bars ──────────────────────────────
        ax = ax1
        n_org = len(organisms)
        bar_w = 0.20
        x_pos = np.arange(n_org)
        for hi, head in enumerate(head_order):
            means = []
            stds  = []
            for org in organisms:
                m, s = summary.get((head, org), (np.nan, 0.0))
                means.append(m); stds.append(s)
            offset = (hi - 1.5) * bar_w
            ax.bar(x_pos + offset, means, width=bar_w, yerr=stds,
                   color=head_colour[head],
                   alpha=(0.92 if head == "biology" else 0.72),
                   capsize=2, lw=0.5, edgecolor="#444",
                   label=head_display[head])
        ax.axhline(0.5, color="#555", lw=0.7, ls="--", alpha=0.7)
        ax.text(n_org - 0.55, 0.51, "chance",
                ha="right", va="bottom", fontsize=6.5, color="#555",
                fontstyle="italic")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{org_display[o]}\n{org_gc[o]}\\,\\% GC" for o in organisms],
            fontsize=7.5)
        ax.set_ylim(0.30, 1.0)
        ax.set_ylabel(r"AUROC (3 seeds, train$=$\textit{S.\,cer.})",
                      fontsize=8)
        ax.set_title(r"cls collapses universally OOD; "
                     r"phase-INV transfers; biology lifts further",
                     fontsize=9, pad=6)
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=6.8,
                  handlelength=1.2, columnspacing=0.8, borderpad=0.2)

        # ── Panel 3: Δ biology − cls per species ──────────────────────────
        ax = ax2
        deltas = []
        for org in organisms:
            # Use the better of the two cls baselines (sinusoidal or APE)
            # so the gap is the conservative one — biology vs the strongest
            # cls competitor.
            m_sin, _ = summary.get(("cls_sin", org), (np.nan, 0.0))
            m_ape, _ = summary.get(("cls_ape", org), (np.nan, 0.0))
            cls_candidates = [v for v in (m_sin, m_ape) if not np.isnan(v)]
            m_cls = max(cls_candidates) if cls_candidates else np.nan
            m_bio, _ = summary.get(("biology", org), (np.nan, 0.0))
            deltas.append(m_bio - m_cls)
        bar_colours = [C_SIN if d > 0 else B_OOD for d in deltas]
        ax.bar(x_pos, deltas, width=0.7,
               color=bar_colours, alpha=0.85,
               lw=0.5, edgecolor="#444")
        ax.axhline(0.0, color="#aaa", lw=0.5)
        # Synthetic hero reference (0.66 = 1.00 - 0.34)
        ax.axhline(0.66, color=C_POLY, lw=0.7, ls=":", alpha=0.8)
        ax.text(n_org - 0.5, 0.665, "synthetic hero $\\Delta\\!=\\!0.66$",
                ha="right", va="bottom", fontsize=6.5, color=C_POLY,
                fontstyle="italic")
        # Annotate values
        for xi, d in enumerate(deltas):
            ax.text(xi, d + 0.015, f"$+{d:.2f}$", ha="center", va="bottom",
                    fontsize=7, color="#222")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([org_display[o] for o in organisms],
                           rotation=20, ha="right", fontsize=7.5)
        ax.set_ylim(0.0, 0.72)
        ax.set_ylabel(r"$\Delta_\mathrm{biology - cls}$", fontsize=8)
        ax.set_title(r"Catastrophic-collapse magnitude on real DNA",
                     fontsize=9, pad=6)

        fig.tight_layout()
        out = os.path.join(output_dir, "exp_cds_species_concept.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="figures")
    parser.add_argument("--exp", default=None,
                        help=f"Plot one concept figure only ({list(DISPATCH)})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp:
        if args.exp not in DISPATCH:
            print(f"Unknown experiment: {args.exp}. Choose from {list(DISPATCH)}")
            return
        DISPATCH[args.exp](args.output_dir)
    else:
        for key, fn in DISPATCH.items():
            print(f"Plotting Exp {key} concept...")
            fn(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
