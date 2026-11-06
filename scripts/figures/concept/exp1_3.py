"""Exp 1.3 concept figure: G_1(x) = sin(x) cos(2x) on the SINDy two-level battery.

Mirrors Exp 1.1's layout:
  Left   - G_1(x) = sin(x) cos(2x) across training and OOD zones
  Centre - raw feature space: two disjoint regions on R
  Right  - 2-D Fourier feature space (sin x, sin 2x): training and OOD scatter
           overlap on the same compact region (extrapolation = interpolation)

Run as a script:
    conda run -n data_analysis python scripts/plot_concept.py --exp 1.3
"""

# %% imports
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _helpers import shade_zone
from _style import (B_OOD, B_TRAIN, C_OOD, C_SIN, C_TRAIN, RC)

OUT = Path(__file__).resolve().parents[3] / "figures" / "exp1_3_concept.pdf"

X_TR_LO, X_TR_HI = -np.pi, np.pi
X_OOD_LO, X_OOD_HI = 2 * np.pi, 4 * np.pi


def _G1(x):
    return np.sin(x) * np.cos(2 * x)


# %% render
def render():
    x_full = np.linspace(X_TR_LO - 0.15, X_OOD_HI + 0.15, 1600)

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

        # Panel 1: DGP curve
        ax = ax_left
        shade_zone(ax, X_TR_LO,  X_TR_HI,  C_TRAIN, B_TRAIN, "Training")
        # gap region [pi, 2pi] left unshaded
        shade_zone(ax, X_OOD_LO, X_OOD_HI, C_OOD,   B_OOD,   "OOD")

        ax.plot(x_full, _G1(x_full), color=C_SIN, lw=2.0,
                label=r"$G_1(x)=\sin(x)\cos(2x)$", zorder=3)

        ax.set_xlim(X_TR_LO - 0.15, X_OOD_HI + 0.15)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$G_1(x)$", fontsize=10)
        ax.set_xticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        ax.set_xticklabels(
            [r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"],
            fontsize=10,
        )

        ax.text((X_TR_LO + X_TR_HI) / 2,  1.10, "ID (train)", ha="center",
                fontsize=10, color=B_TRAIN, fontweight="bold")
        ax.text((X_OOD_LO + X_OOD_HI) / 2, 1.10, "OOD", ha="center",
                fontsize=10, color=B_OOD,   fontweight="bold")

        # Highlight that G_1 = (sin 3x - sin x)/2 contains a frequency
        # that is NOT in the {sin x, cos x, sin 2x, cos 2x} basis used below
        ax.annotate(r"$G_1=\frac{1}{2}(\sin 3x-\sin x)$",
                    xy=(0, _G1(0)),
                    xytext=(np.pi / 2, 0.85), fontsize=10, ha="center",
                    fontweight="bold",
                    color="black",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6))

        dgp_handles = [
            mlines.Line2D([], [], color=C_SIN, lw=2,
                          label=r"$G_1(x)=\sin(x)\cos(2x)$"),
        ]
        ax.legend(handles=dgp_handles, fontsize=10, frameon=False,
                  loc="lower right")
        ax.set_title("DGP across zones", fontsize=10, fontweight="bold",
                     pad=14)

        # Panel 2: raw-feature-space strip
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

        draw_segment(ax, 0.00, 0.36, 0.55, C_TRAIN, B_TRAIN,
                     r"$[-\pi,\pi]$ ID", B_TRAIN)
        draw_segment(ax, 0.62, 1.00, 0.55, C_OOD,   B_OOD,
                     r"$[2\pi,4\pi]$ OOD", B_OOD)

        ax.annotate("", xy=(0.61, 0.55), xytext=(0.37, 0.55),
                    arrowprops=dict(arrowstyle="<->", color="grey", lw=1))

        ax.text(0.5, -0.5, "disjoint: must extrapolate",
                fontweight='bold',
                ha="center", fontsize=10, color="black", style="italic")

        # Panel 3: 2-D Fourier feature scatter (sin x, sin 2x)
        ax = ax_right
        ax.set_aspect("equal")
        ax.set_xlim(-1.65, 1.45)
        ax.set_ylim(-1.55, 1.30)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_title("Transformed Space",
                     fontsize=10, fontweight="bold", pad=14)
        ax.text(0.5, 1.0,
                r"Fourier $\varphi(x){=}(\sin x,\sin 2x)$: same compact region",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=9)

        # Continuous lemniscate-shaped curve in (sin x, sin 2x), period 2*pi
        x_curve = np.linspace(0, 2 * np.pi, 800)
        ax.plot(np.sin(x_curve), np.sin(2 * x_curve),
                color="lightgrey", lw=1.0, zorder=1)

        rng = np.random.default_rng(42)
        x_train_pts = rng.uniform(X_TR_LO, X_TR_HI, 360)
        x_ood_pts   = rng.uniform(X_OOD_LO, X_OOD_HI, 360)

        ax.scatter(np.sin(x_train_pts), np.sin(2 * x_train_pts),
                   s=5, alpha=0.4, color=B_TRAIN,
                   label=r"ID $[-\pi,\pi]$",   zorder=3)
        ax.scatter(np.sin(x_ood_pts),   np.sin(2 * x_ood_pts),
                   s=5, alpha=0.4, color=B_OOD, marker="x",
                   label=r"OOD $[2\pi,4\pi]$", zorder=2)

        ax.set_xlabel(r"$\sin(x)$", fontsize=10)
        ax.set_ylabel(r"$\sin(2x)$", fontsize=10)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

        ax.text(0.5, -0.30,
                "OOD lands in the same lemniscate: \n extrapolation = interpolation",
                fontweight='bold',
                ha="center", va="top", transform=ax.transAxes,
                fontsize=10, color="black", style="italic")

        ax.legend(fontsize=9, frameon=False, loc="upper left",
                  bbox_to_anchor=(.8, 0.98))

        # Frame each right-column panel + connecting arrow
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def _fig_frac(bb):
            return bb.transformed(fig.transFigure.inverted())

        bb_orig  = _fig_frac(ax_center.get_tightbbox(renderer))
        bb_trans = _fig_frac(ax_right.get_tightbbox(renderer))

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

        arrow_top    = bb_orig.y0 - pad_y - 0.006
        arrow_bottom = bb_trans.y1 + pad_y + 0.006
        if arrow_top - arrow_bottom < 0.04:
            arrow_bottom = arrow_top - 0.04
        arrow = mpatches.FancyArrowPatch(
            (cx, arrow_top),
            (cx, arrow_bottom),
            arrowstyle="-|>", mutation_scale=14,
            color="#444", lw=1.4,
            transform=fig.transFigure, zorder=5, clip_on=False,
        )
        fig.patches.append(arrow)

        # Stretch the left panel to match the right column's full tightbbox
        # height (frames + outer italic text). Without this, the right-column
        # frames + sub-panel labels extend past the left panel's tightbbox so
        # bbox_inches="tight" leaves whitespace above and below the left panel.
        fig.canvas.draw()
        inv_fig = fig.transFigure.inverted()
        bb_orig_full  = ax_center.get_tightbbox(renderer).transformed(inv_fig)
        bb_trans_full = ax_right.get_tightbbox(renderer).transformed(inv_fig)
        right_top_frac    = max(bb_orig_full.y1, bb_trans_full.y1) + pad_y
        right_bottom_frac = min(bb_orig_full.y0, bb_trans_full.y0) - pad_y

        bb_left_tight = ax_left.get_tightbbox(renderer).transformed(inv_fig)
        left_pos = ax_left.get_position()
        top_overflow    = bb_left_tight.y1 - left_pos.y1
        bottom_overflow = left_pos.y0      - bb_left_tight.y0

        new_y1 = right_top_frac    - top_overflow
        new_y0 = right_bottom_frac + bottom_overflow
        ax_left.set_position([left_pos.x0, new_y0,
                              left_pos.width, new_y1 - new_y0])
    return fig


# %% build
def build(output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = render()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved -> {out}")
    return out


if __name__ == "__main__":
    fig = render()
    fig
# fig = render()
# fig
# %%
