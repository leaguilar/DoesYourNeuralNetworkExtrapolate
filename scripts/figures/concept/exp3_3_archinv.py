"""Hero / architecture-invariance concept figure.

Split into two PDFs so they can be stacked in the paper:
  - exp3_3_concept_ab.pdf  -- (a) schematic + (b) full-scale sequence
  - exp3_3_concept_cd.pdf  -- (c) S^1 geometry + (d) OOD bar chart

Run as a script:
    conda run -n data_analysis python scripts/plot_concept.py --exp hero
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _helpers import shade_zone
from _style import (B_OOD, B_TRAIN, C_OOD, C_POLY, C_SIN, C_TRAIN, RC)

OUT_DIR = Path(__file__).resolve().parents[3] / "figures"
OUT_AB = OUT_DIR / "exp3_3_concept_ab.pdf"
OUT_CD = OUT_DIR / "exp3_3_concept_cd.pdf"


# %% data setup
def _setup():
    P = 64
    L_train = 256   # = 4P: training covers exactly 4 revolutions of S^1
    L_ood = 128

    rng = np.random.default_rng(0)
    f_map = rng.integers(0, 256, size=P)
    t_all = np.arange(L_train + L_ood)
    y_all = f_map[t_all % P] / 255.0
    x_rand = rng.integers(0, 256, size=L_train + L_ood) / 255.0

    P_vis = 6
    y_pattern = ['w', 's', 'd', 'd', 'f', 'a']
    N_TRAIN_VIS = 2 * P_vis
    N_OOD_VIS = 2 * P_vis
    rng_vis = np.random.default_rng(7)
    x_chars = [c for c in 'bceghjklmnopqrtuvxyz' if c not in y_pattern]
    x_train_str = [x_chars[i % len(x_chars)] for i in
                   rng_vis.integers(0, len(x_chars), N_TRAIN_VIS)]
    x_ood_str = [x_chars[i % len(x_chars)] for i in
                 rng_vis.integers(0, len(x_chars), N_OOD_VIS)]
    y_train_str = [y_pattern[i % P_vis] for i in range(N_TRAIN_VIS)]
    y_ood_str = [y_pattern[i % P_vis] for i in range(N_OOD_VIS)]

    return dict(
        P=P, L_train=L_train, L_ood=L_ood,
        t_all=t_all, y_all=y_all, x_rand=x_rand,
        P_vis=P_vis, N_TRAIN_VIS=N_TRAIN_VIS, N_OOD_VIS=N_OOD_VIS,
        x_train_str=x_train_str, x_ood_str=x_ood_str,
        y_train_str=y_train_str, y_ood_str=y_ood_str,
    )


# %% panel renderers
def _draw_panel_a(ax, d):
    ax.axis("off")

    CW, CH = 1.0, 0.62
    X0, GAP = 2.4, 1.8
    X_OOD_START = X0 + d["N_TRAIN_VIS"] * CW + GAP
    SEP_X = X0 + d["N_TRAIN_VIS"] * CW + GAP * 0.38
    Y_INPUT, Y_TARGET = 2.1, 0.8

    total_w = X_OOD_START + d["N_OOD_VIS"] * CW + 0.5
    ax.set_xlim(0, total_w)
    ax.set_ylim(-0.5, 4.1)

    def draw_token_row(tokens, x_start, y_ctr, fc, tc):
        for i, tok in enumerate(tokens):
            x = x_start + i * CW
            r = mpatches.FancyBboxPatch(
                (x + 0.06, y_ctr - CH / 2), CW - 0.12, CH,
                boxstyle="round,pad=0.03", fc=fc, ec="white",
                lw=0.35, zorder=3)
            ax.add_patch(r)
            ax.text(x + CW / 2, y_ctr, tok, ha="center", va="center",
                    fontsize=7, color=tc, fontweight="bold",
                    fontfamily="monospace", zorder=4)

    draw_token_row(d["x_train_str"], X0,          Y_INPUT, "#d0d0d0", "#444444")
    draw_token_row(d["x_ood_str"],   X_OOD_START, Y_INPUT, "#d0d0d0", "#444444")
    draw_token_row(d["y_train_str"], X0,          Y_TARGET, B_TRAIN, "white")
    draw_token_row(d["y_ood_str"],   X_OOD_START, Y_TARGET, B_OOD,   "white")

    for y_c in [Y_INPUT, Y_TARGET]:
        ax.text(SEP_X + GAP * 0.30, y_c, "···",
                ha="center", va="center", fontsize=11, color="#888888", zorder=4)

    ax.text(X0 - 0.15, Y_INPUT,  r"$x_t$:", ha="right", va="center",
            fontsize=8, color="#666666")
    ax.text(X0 - 0.15, Y_TARGET, r"$y_t$:", ha="right", va="center",
            fontsize=8, color=B_TRAIN, fontweight="bold")

    ax.plot([SEP_X, SEP_X], [-0.15, 3.75],
            color="#999999", lw=0.9, ls="--", zorder=2)

    X_TRAIN_MID = X0 + d["N_TRAIN_VIS"] * CW / 2
    X_OOD_MID = X_OOD_START + d["N_OOD_VIS"] * CW / 2
    ax.text(X_TRAIN_MID, 3.85, "training", ha="center", fontsize=7.5,
            color=B_TRAIN, fontweight="bold")
    ax.text(X_OOD_MID,   3.85, "OOD",      ha="center", fontsize=7.5,
            color=B_OOD,   fontweight="bold")

    P_vis = d["P_vis"]
    brace_y = Y_TARGET + CH / 2 + 0.22
    for k in range(2):
        xL = X0 + k * P_vis * CW + 0.06
        xR = xL + P_vis * CW - 0.18
        ax.annotate("", xy=(xR, brace_y), xytext=(xL, brace_y),
                    arrowprops=dict(arrowstyle="|-|", color=B_TRAIN, lw=0.8,
                                    mutation_scale=3))
        ax.text((xL + xR) / 2, brace_y + 0.18, r"$P_\mathrm{vis}$",
                ha="center", fontsize=6, color=B_TRAIN)
    for k in range(2):
        xL = X_OOD_START + k * P_vis * CW + 0.06
        xR = xL + P_vis * CW - 0.18
        ax.annotate("", xy=(xR, brace_y), xytext=(xL, brace_y),
                    arrowprops=dict(arrowstyle="|-|", color=B_OOD, lw=0.8,
                                    mutation_scale=3))

    ax.text(X_TRAIN_MID, -0.18,
            r"$x_t$: random (no position info)",
            ha="center", fontsize=6.5, color="#666666", style="italic")
    ax.text(X_TRAIN_MID, -0.46,
            r"$y_t = f(t\,\mathrm{mod}\,P)$: periodic pattern continues OOD",
            ha="center", fontsize=6.5, color=B_TRAIN, style="italic")

    ax.set_title(r"Sequence structure (schematic, $P_\mathrm{vis}{=}6$;  actual $P{=}64$)",
                 fontsize=8)


def _draw_panel_b(ax, d):
    P, L_train, L_ood = d["P"], d["L_train"], d["L_ood"]
    t_all, y_all, x_rand = d["t_all"], d["y_all"], d["x_rand"]

    shade_zone(ax, 0,       L_train,         C_TRAIN, B_TRAIN, "Training")
    shade_zone(ax, L_train, L_train + L_ood, C_OOD,   B_OOD,   "OOD")

    ax.scatter(t_all[::3], x_rand[::3], color="grey", s=3, alpha=0.35, zorder=2)

    ax.plot(t_all[:L_train], y_all[:L_train], color=B_TRAIN, lw=1.5, zorder=3)
    ax.plot(t_all[L_train:], y_all[L_train:], color=B_OOD,   lw=1.5, zorder=3)

    for k in range(0, L_train + L_ood, P):
        ax.axvline(k, color="lightgrey", lw=0.5, ls=":", zorder=1)

    ax.set_xlim(-4, L_train + L_ood + 4)
    ax.set_ylim(-0.06, 1.10)
    ax.set_xlabel(r"Position $t$")
    ax.set_ylabel("Token value (norm.)")
    ax.set_xticks([0, P, 2 * P, 3 * P, L_train, L_train + P, L_train + 2 * P])
    ax.set_xticklabels(["$0$", "$P$", "$2P$", "$3P$",
                        r"$L_\mathrm{tr}$", r"$L_\mathrm{tr}{+}P$",
                        r"$L_\mathrm{tr}{+}2P$"], fontsize=6.5)
    ax.text(L_train / 2,         1.07, "train", ha="center", fontsize=7,
            color=B_TRAIN, fontweight="bold")
    ax.text(L_train + L_ood / 2, 1.07, "OOD",   ha="center", fontsize=7,
            color=B_OOD, fontweight="bold")
    legend_handles_p2 = [
        mlines.Line2D([], [], color="grey", marker="o", lw=0, ms=4, alpha=0.6,
                      label=r"$x_t\!\sim\!\mathrm{Uniform}$ (random)"),
        mlines.Line2D([], [], color=B_TRAIN, lw=1.5,
                      label=r"$y_t{=}f(t\,\mathrm{mod}\,P)$ (periodic)"),
    ]
    ax.legend(handles=legend_handles_p2, fontsize=6, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handlelength=1.6, handletextpad=0.5)
    ax.set_title(r"Full scale: $P{=}64$, $L_\mathrm{tr}{=}256$")


def _draw_panel_c(ax, d):
    P, L_train, L_ood = d["P"], d["L_train"], d["L_ood"]

    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.2)
    ax.axis("off")
    ax.set_title(r"$\varphi(t)=(\sin 2\pi t/P,\,\cos 2\pi t/P)\in S^1$",
                 pad=14)

    theta_c = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta_c), np.sin(theta_c), color="lightgrey", lw=0.8, zorder=1)

    t_tr = np.linspace(0, L_train, 3000)
    ang_tr = 2 * np.pi * t_tr / P
    ax.plot(np.cos(ang_tr), np.sin(ang_tr), color=B_TRAIN, lw=5, alpha=0.55,
            solid_capstyle="round", zorder=3)

    t_od = np.linspace(L_train, L_train + L_ood, 800)
    ang_od = 2 * np.pi * t_od / P
    r_od = 1.45
    ax.plot(np.cos(ang_od) * r_od, np.sin(ang_od) * r_od,
            color=B_OOD, lw=1.8, ls="--", alpha=0.75, zorder=2)

    j = 20
    ang_j = 2 * np.pi * j / P
    cx, cy = np.cos(ang_j), np.sin(ang_j)
    ox, oy = cx * r_od, cy * r_od
    ax.scatter([cx], [cy], s=55, color=B_TRAIN, zorder=6)
    ax.scatter([ox], [oy], s=55, color=B_OOD,   zorder=6, marker="D")

    ax.annotate("",
                xy=(cx * 1.06, cy * 1.06),
                xytext=(ox * 0.96, oy * 0.96),
                arrowprops=dict(arrowstyle="-|>", color=C_SIN, lw=1.6))

    ax.text(ox - 0.55, oy + 0.05,
            rf"$t={L_train + j}$" + "\n(OOD)", ha="right", va="center",
            fontsize=6.5, color=B_OOD)
    ax.text(cx * 0.50, cy * 0.50,
            rf"$t={j}$" + "\n(train)", ha="center", fontsize=6.5, color=B_TRAIN)

    ax.text(0, 2.05,
            r"$L_\mathrm{tr}{=}4P$: training tiles $S^1$ completely",
            ha="center", fontsize=6.5, color=B_TRAIN)

    handles2 = [
        mpatches.Patch(fc=C_TRAIN, ec=B_TRAIN, lw=0.8, label="Train (4 revolutions)"),
        mlines.Line2D([], [], color=B_OOD, lw=1.5, ls="--", label="OOD (same circle)"),
        mlines.Line2D([], [], color=C_SIN, lw=1.5, marker=">", ms=5,
                      label=r"$\varphi(L_\mathrm{tr}{+}j)=\varphi(j)$: OOD $\to$ interpolation"),
    ]
    ax.legend(handles=handles2, fontsize=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), handlelength=1.6, handletextpad=0.5)


def _draw_panel_d(ax, d):
    pe_labels = [
        "Exact\nFourier",
        "RoPE",
        "Sinusoidal",
        "Learned\nAPE",
        "No PE",
    ]
    ood_acc = [100.00, 8.23, 1.56, 0.72, 1.59]
    ood_vals = [100.0 - v for v in ood_acc]
    bar_colors = [C_SIN, "#8888aa", "#aaaaaa", C_POLY, "#cccccc"]
    chance_pct = 100.0 - 100.0 / 256

    bars = ax.bar(range(len(pe_labels)), ood_vals,
                  color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(chance_pct, color=B_OOD, lw=1.1, ls="--", zorder=4)
    ax.text(0.0, chance_pct + 1.5,
            f"chance {chance_pct:.1f}%", ha="left", fontsize=6, color=B_OOD)

    for bar, v in zip(bars, ood_vals):
        lbl = f"{v:.0f}%" if v >= 5 else f"{v:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2, lbl,
                ha="center", va="bottom", fontsize=6.5,
                fontweight="bold" if v < 50 else "normal")

    ax.set_xticks(range(len(pe_labels)))
    ax.set_xticklabels(pe_labels, fontsize=7)
    ax.set_ylabel("OOD error (%)")
    ax.set_ylim(0, 115)
    ax.set_title("OOD error (ID error: 0%\nfor all trained models)")


def _add_panel_label(ax, label):
    ax.text(0.0, 1.10, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


# %% render — pure: builds and returns figs, no save, no close
def render_ab():
    d = _setup()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(12.0, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.45], wspace=0.12)
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])
        _add_panel_label(ax_a, "(a)")
        _add_panel_label(ax_b, "(b)")
        _draw_panel_a(ax_a, d)
        _draw_panel_b(ax_b, d)
    return fig


def render_cd():
    d = _setup()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(9.0, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.10], wspace=0.10)
        ax_c = fig.add_subplot(gs[0], aspect="equal")
        ax_d = fig.add_subplot(gs[1])
        _add_panel_label(ax_c, "(c)")
        _add_panel_label(ax_d, "(d)")
        _draw_panel_c(ax_c, d)
        _draw_panel_d(ax_d, d)
    return fig


# %% build — CLI / shell entry point
def build(output_dir=None):
    out_dir = Path(output_dir) if output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname, render_fn in [(OUT_AB.name, render_ab), (OUT_CD.name, render_cd)]:
        out = out_dir / fname
        with plt.rc_context(RC):
            fig = render_fn()
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
        print(f"  Saved -> {out}")
        paths.append(out)
    return paths


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    fig_ab = render_ab()
    fig_cd = render_cd()
    fig_ab
