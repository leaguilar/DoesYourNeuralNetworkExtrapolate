"""Mass-action-kinetics vector-field figure (Exp MAK-VF).

Two panels mirroring Fig 1 of Vasiliauskaite & Antulov-Fantulin (2024):
  Left   raw inputs (x_1, x_2): MLP predictions deviate from the true field OOD
  Right  bilinear features [x_i, x_i x_j]: OLS overlays the true field

Loads grid-evaluated predictions from
    results/exp2_1/figure_artefacts.npz
which is written by src/experiments/exp2_4_mak_dynamics.py when invoked
with --figure.

Run as a script:
    conda run -n data_analysis python scripts/main_plot_concept.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import RC

OUT = Path(__file__).resolve().parents[3] / "figures" / "main" / "exp2_1_mak.pdf"
ARTEFACTS = (Path(__file__).resolve().parents[3]
             / "results" / "exp2_1" / "figure_artefacts.npz")


# %% data loader
def _load_artefacts(path=None):
    """Read grid-evaluated MAK vector fields written by the experiment."""
    p = Path(path or ARTEFACTS)
    if not p.is_file():
        return None
    z = np.load(p)
    return dict(
        x1_tr=z["x1_tr"], x2_tr=z["x2_tr"],
        g=z["g"],
        U_t=z["U_t"], V_t=z["V_t"],
        U_r=z["U_r"], V_r=z["V_r"],
        U_o=z["U_o"], V_o=z["V_o"],
        state_max=z["state_max"],
        train_center=z["train_center"],
        train_half=z["train_half"],
    )


# %% render — pure: builds and returns fig (or None when no data), no save, no close
def render(data=None):
    if data is None:
        data = _load_artefacts()
    if data is None:
        print(f"  [exp2_1_mak] no artefacts at {ARTEFACTS} — skipping")
        return None

    g           = np.asarray(data["g"])
    U_t, V_t    = np.asarray(data["U_t"]), np.asarray(data["V_t"])
    U_r, V_r    = np.asarray(data["U_r"]), np.asarray(data["V_r"])
    U_o, V_o    = np.asarray(data["U_o"]), np.asarray(data["V_o"])
    x1_tr       = np.asarray(data["x1_tr"])
    x2_tr       = np.asarray(data["x2_tr"])
    state_max   = float(data["state_max"])

    BLUE = "#2196F3"
    PINK = "#E91E8C"
    TICKS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]

    n_seeds = 7
    sp = np.linspace(0.1, state_max - 0.1, n_seeds)
    SX, SY = np.meshgrid(sp, sp)
    start_pts = np.column_stack([SX.ravel(), SY.ravel()])

    STRM = dict(
        start_points=start_pts,
        linewidth=0.7,
        arrowsize=0.7,
        integration_direction="both",
        maxlength=3.5,
    )

    rc = {**RC, "axes.grid": False}

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5),
                                 constrained_layout=True)

        panels = [
            (axes[0], U_r, V_r, "raw inputs",          "pred deviates OOD"),
            (axes[1], U_o, V_o, r"bilinear $\varphi$", "pred overlays true (hidden)"),
        ]

        for ax, U_pred, V_pred, panel_label, panel_note in panels:
            ax.set_xlim(0, state_max)
            ax.set_ylim(0, state_max)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_xticks(TICKS)
            ax.set_yticks(TICKS)

            ax.streamplot(g, g, U_pred, V_pred, color=PINK, zorder=2, **STRM)
            ax.streamplot(g, g, U_t,    V_t,    color=BLUE, zorder=3, **STRM)

            ax.scatter(x1_tr, x2_tr, c="black", s=7, marker="+",
                       linewidths=0.6, alpha=0.5, zorder=5)

            label_bbox = dict(facecolor="white", alpha=0.75,
                              edgecolor="none", boxstyle="round,pad=0.25")
            ax.text(0.03, 0.97, panel_label, transform=ax.transAxes,
                    fontsize=13.5, fontweight="bold", va="top", ha="left",
                    bbox=label_bbox, zorder=10)
            ax.text(0.03, 0.88, panel_note, transform=ax.transAxes,
                    fontsize=10.5, va="top", ha="left",
                    color="#444", style="italic",
                    bbox=label_bbox, zorder=10)

        legend_elements = [
            Line2D([0], [0], color=BLUE, lw=1.5, label="True"),
            Line2D([0], [0], color=PINK, lw=1.5, label="Predicted"),
            Line2D([0], [0], color="black", marker="+", lw=0,
                   markersize=7, markeredgewidth=0.9, label="Train"),
        ]
        axes[0].legend(handles=legend_elements,
                       fontsize=9,
                       loc="upper right", framealpha=0.85,
                       borderpad=0.5, handlelength=1.5)
    return fig


# %% build — CLI / shell entry point
def build(output_dir=None, data=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = render(data)
    if fig is None:
        return None
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    fig = render()
    fig
# %%
