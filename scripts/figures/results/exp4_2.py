"""Tor2 coverage: OOD RMSE vs M for d ∈ {4,6,8} under correct φ.

Run as a script:
    conda run -n data_analysis python scripts/plot_results.py --exp tor2_coverage
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _data import load_results, tor_aggseeds
from _style import RC

EID = "tor2_coverage"
OUT = Path(__file__).resolve().parents[3] / "figures" / "exp4_2.pdf"
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

    by_d_M_r2   = defaultdict(lambda: defaultdict(list))
    by_d_M_rmse = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for cell in run["cells"].values():
            if cell.get("status") != "ok":
                continue
            by_d_M_r2[cell["d"]][cell["M"]].append(cell["ood_r2"])
            by_d_M_rmse[cell["d"]][cell["M"]].append(cell["ood_rmse"])

    ds = sorted(by_d_M_r2.keys())
    if not ds:
        print("  [exp4_2] no successful cells — skipping plot")
        return None

    cmap = plt.get_cmap("viridis")
    d_markers = ["o", "s", "^", "D", "v"]
    d_linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.0))
        fig.text(0.005, 0.005, "(c)", fontsize=9, fontweight="bold",
                 va="bottom", ha="left")
        all_max = 0.0
        for i, d in enumerate(ds):
            color = cmap(0.15 + 0.7 * i / max(1, len(ds) - 1))
            marker = d_markers[i % len(d_markers)]
            ls = d_linestyles[i % len(d_linestyles)]
            Ms = sorted(by_d_M_rmse[d].keys())
            rm = np.array([tor_aggseeds(by_d_M_rmse[d][M])[0] for M in Ms])
            rs = np.array([tor_aggseeds(by_d_M_rmse[d][M])[1] for M in Ms])
            all_max = max(all_max, float((rm + rs).max()))
            ax.plot(Ms, rm, marker=marker, markersize=3.5, color=color,
                    ls=ls, lw=1.3, label=f"$d={d}$")
            ax.fill_between(Ms, rm - rs, rm + rs, color=color, alpha=0.15, lw=0)
        all_Ms = sorted({M for d in ds for M in by_d_M_rmse[d].keys()})
        ax.set_xlabel(r"Coverage per dimension $M$", labelpad=1)
        ax.set_ylabel("OOD RMSE", labelpad=2)
        ax.set_xticks(all_Ms)
        ax.set_ylim(0, all_max * 1.08)
        ax.tick_params(axis="both", pad=2)
        ax.legend(loc="upper right", frameon=False, handlelength=1.2,
                  handletextpad=0.4, borderpad=0.2, labelspacing=0.2,
                  fontsize=9.6)
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
