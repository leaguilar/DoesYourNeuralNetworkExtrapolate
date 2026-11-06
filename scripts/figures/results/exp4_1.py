"""Tor1 representation: OOD RMSE vs d for raw / learned-Fourier / exact-φ MLP.

Run as a script:
    conda run -n data_analysis python scripts/plot_results.py --exp tor1_representation
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
from _style import RC, TOR_C_EXACT, TOR_C_LEARN, TOR_C_RAW

EID = "tor1_representation"
OUT = Path(__file__).resolve().parents[3] / "figures" / "exp4_1.pdf"
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

    by_d = defaultdict(lambda: {"raw_mlp": [], "learned_fourier_mlp": [], "exact_phi_mlp": []})
    for run in runs:
        for cell in run["conditions"].values():
            d = cell["d"]
            for cond in ["raw_mlp", "learned_fourier_mlp", "exact_phi_mlp"]:
                by_d[d][cond].append(cell[cond]["ood_rmse"])
    ds = sorted(by_d.keys())
    means, stds = {}, {}
    for cond in ("raw_mlp", "learned_fourier_mlp", "exact_phi_mlp"):
        means[cond] = []
        stds[cond] = []
        for d in ds:
            m, s = tor_aggseeds(by_d[d][cond])
            means[cond].append(m); stds[cond].append(s)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.0))
        fig.text(0.005, 0.005, "(a)", fontsize=9, fontweight="bold",
                 va="bottom", ha="left")
        series = [
            ("raw_mlp",             "Raw",                TOR_C_RAW,   "o", ":"),
            ("learned_fourier_mlp", "Learned",            TOR_C_LEARN, "s", "--"),
            ("exact_phi_mlp",       r"Correct $\varphi$", TOR_C_EXACT, "^", "-"),
        ]
        for cond, label, c, marker, ls in series:
            m = np.array(means[cond]); s = np.array(stds[cond])
            ax.plot(ds, m, label=label, color=c, marker=marker, ls=ls, lw=1.3, ms=3.5)
            ax.fill_between(ds, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.set_xlabel(r"Periodic dimension $d$", labelpad=1)
        ax.set_ylabel("OOD RMSE", labelpad=2)
        ax.legend(loc="upper right", frameon=False, handlelength=1.4,
                  handletextpad=0.4, borderpad=0.2, labelspacing=0.2,
                  fontsize=9.6)
        ax.set_xticks(ds)
        ax.tick_params(axis="both", pad=2)
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
