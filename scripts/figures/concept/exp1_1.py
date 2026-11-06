"""Exp 1.1 concept figure: sin(x) vs Taylor-9 across training and OOD zones.

Three panels:
  Left   – sin(x) vs Taylor-9 across all three zones
  Centre – raw feature space: three disjoint regions on R
  Right  – Fourier feature space S^1: training (left arc) and near-boundary
           (right arc) together tile the circle; OOD maps onto the same circle

Run as a script:
    conda run -n data_analysis python scripts/plot_concept.py --exp 1.1
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _helpers import shade_zone, taylor9
from _style import (B_OOD, B_TRAIN, C_OOD, C_POLY, C_SIN, C_TRAIN, RC)

OUT = Path(__file__).resolve().parents[3] / "figures" / "exp1_1_concept.pdf"
PRED_CACHE = Path(__file__).resolve().parents[3] / "figures" / "_cache" / "exp1_1_predictions.npz"

# "From-raw" baseline overlay colours (not in shared palette — local to this fig)
C_SINDY     = "#9b59b6"   # purple
C_MLP       = "#e67e22"   # orange
C_TABPFN    = "#1abc9c"   # teal
C_TIMESFM   = "#fb03ff"   # dark brown — TimesFM on sin context
C_TIMESFM_T = "#4c2600"   # burnt-orange — TimesFM on Taylor-9 context


def _load_predictions():
    """Return predictions dict or None if cache missing."""
    if not PRED_CACHE.exists():
        return None
    with np.load(PRED_CACHE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# %% render — pure: builds and returns fig, no save, no close
def render():
    x_full = np.linspace(-np.pi - 0.15, 3 * np.pi + 0.15, 1200)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(10, 3.7))
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[1, 5.0],
            width_ratios=[1.55, 1],
            hspace=1.05, wspace=0.04,
            left=0.07, right=0.96, top=0.90, bottom=0.13,
        )
        ax_left   = fig.add_subplot(gs[:, 0])
        ax_center = fig.add_subplot(gs[0, 1])
        ax_right  = fig.add_subplot(gs[1, 1])

        # ── Panel 1: DGP curves ──────────────────────────────────────────
        ax = ax_left
        shade_zone(ax, -np.pi, np.pi,    C_TRAIN, B_TRAIN, "Training")
        shade_zone(ax,  np.pi, 3 * np.pi, C_OOD,   B_OOD,   "OOD")

        y_sin = np.sin(x_full)
        y_t9  = np.clip(taylor9(x_full), -4, 4)

        ax.plot(x_full, y_sin, color=C_SIN,  lw=2.4, label=r"$\sin(x)$",
                zorder=4)
        ax.plot(x_full, y_t9,  color=C_POLY, lw=2.4, ls="--",
                label="Taylor-9", zorder=4)

        # ── From-raw baseline predictions (overlay) ─────────────────────────
        preds = _load_predictions()
        max_train_dev_pct = float(np.abs(y_t9 - y_sin)[
            (x_full >= -np.pi) & (x_full <= np.pi)].max()
            / (np.abs(y_sin[(x_full >= -np.pi) & (x_full <= np.pi)]).mean() + 1e-12) * 100)
        if preds is not None:
            x_d = preds["x_dense"]
            x_o = preds["x_ood"]
            y_clip_lo, y_clip_hi = -4.0, 4.0
            id_mask = (x_d >= -np.pi) & (x_d <= np.pi)
            sin_id  = np.sin(x_d[id_mask])
            yscale  = np.abs(sin_id).mean() + 1e-12
            for key in ("sindy_raw_dense", "mlp_raw_dense", "tabpfn_raw_dense"):
                if key in preds:
                    dev = float(np.abs(preds[key][id_mask] - sin_id).max() / yscale * 100)
                    max_train_dev_pct = max(max_train_dev_pct, dev)
            if "sindy_raw_dense" in preds:
                ax.plot(x_d, np.clip(preds["sindy_raw_dense"], y_clip_lo, y_clip_hi),
                        color=C_SINDY, lw=2.4, ls=":",
                        label="SINDy (raw on sin)", zorder=3)
            if "mlp_raw_dense" in preds:
                ax.plot(x_d, np.clip(preds["mlp_raw_dense"], y_clip_lo, y_clip_hi),
                        color=C_MLP, lw=1.68, ls="-.",
                        label="MLP (raw on sin)", zorder=3)
            if "tabpfn_raw_dense" in preds:
                ax.plot(x_d, np.clip(preds["tabpfn_raw_dense"], y_clip_lo, y_clip_hi),
                        color=C_TABPFN, lw=1.68, ls=(0, (3, 1, 1, 1)),
                        label="TabPFN (raw on sin)", zorder=3)
            if "timesfm_raw_ood" in preds:
                ax.plot(x_o, np.clip(preds["timesfm_raw_ood"], y_clip_lo, y_clip_hi),
                        color=C_TIMESFM, lw=2.4, ls="-",
                        label="TimesFM (raw on sin)", zorder=5)
            if "timesfm_taylor9_ood" in preds:
                ax.plot(x_o, np.clip(preds["timesfm_taylor9_ood"], y_clip_lo, y_clip_hi),
                        color=C_TIMESFM_T, lw=2.4, ls=(0, (1, 1)),
                        label="TimesFM (raw on Taylor-9)", zorder=5)

        ax.set_xlim(-np.pi - 0.15, 3 * np.pi + 0.15)
        ax.set_ylim(-3.8, 3.8)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$f(x)$", fontsize=10)
        ax.set_xticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"], fontsize=10)

        ax.text(-np.pi+1.3,         3.4, "ID (train)", ha="center", fontsize=10,
                color=B_TRAIN, fontweight="bold")
        ax.text(np.pi+0.8, 3.4, "OOD",        ha="center", fontsize=10,
                color=B_OOD,   fontweight="bold")

        ax.annotate(f"max {max_train_dev_pct:.2f}%\ndeviation",
                    xy=(-0.6, np.sin(-0.6)),
                    xytext=(-1.3, 1), fontsize=10, ha="center", color="grey",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

        # Two-box legend:
        #   Box 1 — true DGP sin(x) and the models fit/evaluated on it.
        #   Box 2 — wrong polynomial commitment Taylor-9 and TimesFM evaluated
        #           on the Taylor-9 sequence as context (raw inputs of Taylor-9).
        sin_box_handles = [
            mlines.Line2D([], [], color=C_SIN,     lw=2.4, label=r"$\sin(x)$"),
            mlines.Line2D([], [], color=C_SINDY,   lw=2.4,  ls=":",
                          label="SINDy (raw on sin)"),
            mlines.Line2D([], [], color=C_MLP,     lw=1.68, ls="-.",
                          label="MLP (raw on sin)"),
            mlines.Line2D([], [], color=C_TABPFN,  lw=1.68, ls=(0, (3, 1, 1, 1)),
                          label="TabPFN (raw on sin)"),
            mlines.Line2D([], [], color=C_TIMESFM, lw=2.4,  ls=(0, (1, 1)),
                          label="TimesFM (raw on sin)"),
        ]
        taylor_box_handles = [
            mlines.Line2D([], [], color=C_POLY,      lw=2.4, ls="--",
                          label="Taylor-9"),
            mlines.Line2D([], [], color=C_TIMESFM_T, lw=2.4, ls=(0, (1, 1)),
                          label="TimesFM (raw on Taylor-9)"),
        ]
        leg_sin = ax.legend(handles=sin_box_handles, fontsize=10, frameon=True,
                            loc="lower left", ncol=1, columnspacing=0.8,
                            handlelength=2.4, #title=r"$\sin(x)$ and models on it",
                            title_fontsize=9, borderpad=0.3, labelspacing=0.25)
        leg_sin.get_frame().set_edgecolor("#888")
        leg_sin.get_frame().set_linewidth(0.6)
        ax.add_artist(leg_sin)
        leg_t9 = ax.legend(handles=taylor_box_handles, fontsize=10, frameon=True,
                           loc="lower right", ncol=1, columnspacing=0.8,
                           handlelength=2.4, #title="Taylor-9 and models on it",
                           title_fontsize=9, borderpad=0.3, labelspacing=0.25)
        leg_t9.get_frame().set_edgecolor("#888")
        leg_t9.get_frame().set_linewidth(0.6)
        ax.set_title("DGP pair across zones", fontsize=10, fontweight="bold", pad=14)

        # ── Panel 2: Slim raw-feature-space strip (above right panel) ───
        ax = ax_center
        ax.set_aspect("auto")
        ax.axis("off")
        ax.set_xlim(-0.05, 1.18)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Original Space",
                     fontsize=10, fontweight="bold", pad=14)
        ax.text(0.5, 1.0,
                r"Raw $\mathcal{X}=\mathbb{R}$: disjoint train / OOD",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=9)

        ax.annotate("", xy=(1.13, 0.55), xytext=(-0.03, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1))
        ax.text(1.16, 0.55, r"$x$", va="center", fontsize=10)

        def draw_segment(ax, xL, xR, yc, fc, ec, txt, txt_color):
            h = 0.8
            rect = mpatches.FancyBboxPatch(
                (xL, yc - h / 2), xR - xL, h,
                boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.0, zorder=3)
            ax.add_patch(rect)
            ax.text((xL + xR) / 2, yc, txt, ha="center", va="center",
                    fontsize=9, color=txt_color, zorder=4)

        draw_segment(ax, 0.00, 0.44, 0.55, C_TRAIN, B_TRAIN,
                     r"$[-\pi,\pi]$ ID", B_TRAIN)
        draw_segment(ax, 0.54, 1.00, 0.55, C_OOD,   B_OOD,
                     r"$[\pi,3\pi]$ OOD", B_OOD)

        ax.annotate("", xy=(0.53, 0.55), xytext=(0.45, 0.55),
                    arrowprops=dict(arrowstyle="<->", color="grey", lw=1))

        ax.text(0.5, -0.5, "disjoint: must extrapolate",
                fontweight='bold',
                ha="center", fontsize=10, color="black", style="italic")

        # ── Panel 3: Fourier / S^1 feature space ─────────────────────────
        ax = ax_right
        ax.set_aspect("equal")
        ax.set_xlim(-3.2, 1.4)
        ax.set_ylim(-1.85, 1.4)
        ax.axis("off")
        ax.set_title("Transformed Space",
                     fontsize=10, fontweight="bold", pad=14)
        ax.text(0.5, 1.0,
                r"Fourier $\mathcal{Z}_x=\mathbb{S}^1$: training tiles the circle",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=9)

        cx_shift = -2   # shift circle + inline labels leftward to balance legend

        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(cx_shift + np.cos(theta), np.sin(theta),
                color="lightgrey", lw=1, zorder=1)

        x_tr = np.linspace(-np.pi, np.pi, 600)
        u_tr, v_tr = np.sin(x_tr), np.cos(x_tr)
        ax.plot(cx_shift + u_tr, v_tr, color=B_TRAIN, lw=5, alpha=0.7,
                solid_capstyle="round", zorder=3, label=r"ID $[-\pi,\pi]$")

        x_od = np.linspace(np.pi, 3 * np.pi, 600)
        u_od, v_od = np.sin(x_od), np.cos(x_od)
        ax.plot(cx_shift + u_od * 1.12, v_od * 1.12,
                color=B_OOD, lw=1.8, ls="--", alpha=0.6,
                zorder=2, label=r"OOD $[\pi,3\pi]$")

        ax.annotate("", xy=(cx_shift + 0.95, 0), xytext=(cx_shift - 0.95, 0),
                    arrowprops=dict(arrowstyle="-|>", color=C_SIN, lw=1.4))
        ax.text(cx_shift + 0.0, 0.14, r"$g(u,v)=u$", ha="center", fontsize=9,
                color=C_SIN, fontweight="bold")

        ax.text(cx_shift + 1.0, -1.75,
                "OOD maps to the same circle: \n extrapolation = interpolation",
                fontweight='bold',
                ha="center", fontsize=10, color="black", style="italic")

        ax.legend(fontsize=9, frameon=False, loc="upper right",
                  bbox_to_anchor=(cx_shift+3.15, 1.0))

        # ── Frame each right-column panel + connecting arrow ─────────────
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def _fig_frac(bb):
            return bb.transformed(fig.transFigure.inverted())

        bb_orig  = _fig_frac(ax_center.get_tightbbox(renderer))
        bb_trans = _fig_frac(ax_right.get_tightbbox(renderer))

        # Force both frames to share x-range so their widths are identical
        shared_x0 = min(bb_orig.x0, bb_trans.x0)
        shared_x1 = max(bb_orig.x1, bb_trans.x1)
        bb_orig  = mtransforms.Bbox.from_extents(
            shared_x0, bb_orig.y0,  shared_x1, bb_orig.y1)
        bb_trans = mtransforms.Bbox.from_extents(
            shared_x0, bb_trans.y0, shared_x1, bb_trans.y1)

        pad_x, pad_y = 0.010, 0.010

        def _frame(bb):
            return mpatches.FancyBboxPatch(
                (bb.x0 - pad_x, bb.y0 - pad_y),
                bb.width + 2 * pad_x, bb.height + 2 * pad_y,
                boxstyle="round,pad=0.002,rounding_size=0.010",
                fc="none", ec="#777", lw=0.9,
                transform=fig.transFigure, zorder=0, clip_on=False,
            )

        fig.patches.extend([_frame(bb_orig), _frame(bb_trans)])

        cx = (bb_orig.x0 + bb_orig.x1) / 2

        # Vertical arrow between the two frames
        arrow_top    = bb_orig.y0 - pad_y - 0.006
        arrow_bottom = bb_trans.y1 + pad_y + 0.006
        if arrow_top - arrow_bottom < 0.04:           # minimum visible body
            arrow_bottom = arrow_top - 0.04
        arrow = mpatches.FancyArrowPatch(
            (cx, arrow_top),
            (cx, arrow_bottom),
            arrowstyle="-|>", mutation_scale=14,
            color="#444", lw=1.4,
            transform=fig.transFigure, zorder=5, clip_on=False,
        )
        fig.patches.append(arrow)
    return fig


# %% build — CLI / shell entry point
def build(output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = render()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    fig = render()
    fig
# %%
