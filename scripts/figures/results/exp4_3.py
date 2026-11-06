"""Tor3 capacity: OOD RMSE vs q for d ∈ {4,6,8} with M ∈ {4,5,6} lines.

Run as a script:
    conda run -n data_analysis python scripts/plot_results.py --exp tor3_capacity
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _data import load_results, tor_aggseeds
from _style import RC

EID = "tor3_capacity"
OUT = Path(__file__).resolve().parents[3] / "figures" / "exp4_3.pdf"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"


# %% data loading helper
def _load_runs(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — pure: builds and returns fig (or None when no data), no save, no close
def render(runs=None):
    if runs is None:
        runs = _load_runs()
    if not runs:
        print(f"  [{EID}] no runs found, skipping")
        return None

    by_d_M_q = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        for cell in run["cells"].values():
            if cell.get("status") != "ok":
                continue
            by_d_M_q[cell["d"]][cell["M"]][cell["q"]].append(cell["ood_rmse"])
    ds = sorted(by_d_M_q.keys())
    if not ds:
        print("  [exp4_3] no successful cells — skipping plot")
        return None

    cmap = plt.get_cmap("viridis")
    d_colors = {d: cmap(0.15 + 0.7 * i / max(1, len(ds) - 1)) for i, d in enumerate(ds)}
    d_markers = {d: m for d, m in zip(ds, ["o", "s", "^", "D", "v"])}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.0))
        fig.text(0.005, 0.005, "(b)", fontsize=9, fontweight="bold",
                 va="bottom", ha="left")
        all_max = 0.0
        for d in ds:
            Ms = sorted(by_d_M_q[d].keys())
            for M in Ms:
                qs = sorted(by_d_M_q[d][M].keys())
                rm = np.array([tor_aggseeds(by_d_M_q[d][M][q])[0] for q in qs])
                rs = np.array([tor_aggseeds(by_d_M_q[d][M][q])[1] for q in qs])
                all_max = max(all_max, float((rm + rs).max()))
                color = d_colors[d]
                marker = d_markers[d]
                if M == 4:
                    ax.plot(qs, rm, marker=marker, markersize=3.5, color=color,
                            lw=1.4, label=f"$d={d}$")
                    ax.fill_between(qs, rm - rs, rm + rs, color=color, alpha=0.15, lw=0)
                else:
                    ax.plot(qs, rm, color=color, lw=0.7, alpha=0.45)
        ax.set_xlabel(r"Interaction order $q$", labelpad=1)
        ax.set_ylabel("OOD RMSE", labelpad=2)
        all_qs = sorted({q for d in ds for M in by_d_M_q[d] for q in by_d_M_q[d][M]})
        ax.set_xticks(all_qs)
        ax.set_ylim(0, all_max * 1.08)
        ax.tick_params(axis="both", pad=2)
        d_handles = [
            mlines.Line2D([], [], color=d_colors[d], marker=d_markers[d],
                          ms=3.5, lw=1.4, label=f"$d={d}$")
            for d in ds
        ]
        m_handles = [
            mlines.Line2D([], [], color="gray", lw=1.4, label=r"$M{=}4$ (bold)"),
            mlines.Line2D([], [], color="gray", lw=0.7, alpha=0.45,
                          label=r"$M{\in}\{5,6\}$ (faint)"),
        ]
        leg = ax.legend(handles=d_handles + m_handles, loc="upper left",
                        frameon=False,
                        handlelength=1.2, handletextpad=0.4,
                        borderpad=0.2, labelspacing=0.2, fontsize=9.6)
        for txt in leg.get_texts():
            txt.set_bbox(dict(facecolor="white", alpha=0.75,
                              edgecolor="none", pad=0.5))
        fig.tight_layout(pad=0.3)
    return fig


# %% build — CLI / shell entry point
def build(runs=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = render(runs)
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
