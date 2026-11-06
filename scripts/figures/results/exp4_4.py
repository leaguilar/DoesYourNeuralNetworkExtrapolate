"""Tor-Noise (Exp 4.4): OOD RMSE vs σ for raw / learned-Fourier / exact-φ MLP,
per d ∈ {4, 6, 8}. Single panel; viridis colour + marker per d (matching
exp4_2 / exp4_3); linestyle per commitment.

Run as a script:
    conda run -n data_analysis python scripts/plot_results.py --exp 4.4
"""

# %% imports
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _data import load_results, tor_aggseeds
from _style import RC

_ANALYSIS = Path(__file__).resolve().parents[2] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
from exp4_4_regime_boundary import boundary_interval

EID = "4.4"
OUT = Path(__file__).resolve().parents[3] / "figures" / "exp4_4.pdf"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"

CONDITIONS = ("raw_mlp", "learned_fourier_mlp", "exact_phi_mlp")
COND_LINESTYLE = {
    "raw_mlp":             ":",
    "learned_fourier_mlp": "--",
    "exact_phi_mlp":       "-",
}
COND_LABEL = {
    "raw_mlp":             "Raw",
    "learned_fourier_mlp": "Learned",
    "exact_phi_mlp":       r"Correct $\varphi$",
}


def _load_runs(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


def _aggregate(runs):
    """by_d_sigma[d][sigma][cond] = list of OOD RMSE across seeds."""
    by_d_sigma = defaultdict(lambda: defaultdict(
        lambda: {c: [] for c in CONDITIONS}))
    for run in runs:
        for cell in run["conditions"].values():
            d = int(cell["d"])
            sigma = float(cell["sigma"])
            for cond in CONDITIONS:
                by_d_sigma[d][sigma][cond].append(cell[cond]["ood_rmse"])
    return by_d_sigma


def render(runs=None):
    if runs is None:
        runs = _load_runs()
    if not runs:
        print(f"  [{EID}] no runs found, skipping")
        return None

    by_d_sigma = _aggregate(runs)
    ds = sorted(by_d_sigma.keys())
    if not ds:
        return None

    # Regime boundary σ* = √n ε_W / √(2π), the σ at which the linear form of
    # Prop. 1 stops constraining any test. ε_W is the in-window L² commitment
    # gap, bracketed from the stored cells by
    # scripts/analysis/exp4_4_regime_boundary.py (single source of truth for
    # both this band and the interval quoted in Sec. 6), so the boundary is a
    # band rather than a line.
    _, _, sigma_lo, sigma_hi, _ = boundary_interval(runs)

    cmap = plt.get_cmap("viridis")
    d_colors  = {d: cmap(0.15 + 0.7 * i / max(1, len(ds) - 1)) for i, d in enumerate(ds)}
    d_markers = {d: m for d, m in zip(ds, ["o", "s", "^", "D", "v"])}

    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.0))
        fig.text(0.005, 0.005, "(d)", fontsize=9, fontweight="bold",
                 va="bottom", ha="left")

        for d in ds:
            sigmas_pos = sorted(s for s in by_d_sigma[d].keys() if s > 0)
            color  = d_colors[d]
            marker = d_markers[d]
            for cond in CONDITIONS:
                rm = np.array([tor_aggseeds(by_d_sigma[d][s][cond])[0] for s in sigmas_pos])
                rs = np.array([tor_aggseeds(by_d_sigma[d][s][cond])[1] for s in sigmas_pos])
                ax.plot(sigmas_pos, rm, color=color, marker=marker, markersize=3.0,
                        ls=COND_LINESTYLE[cond], lw=1.2)
                ax.fill_between(sigmas_pos, np.maximum(rm - rs, 1e-6),
                                rm + rs, color=color, alpha=0.10, lw=0)
        ax.axvspan(sigma_lo, sigma_hi, color="0.4", alpha=0.13, lw=0, zorder=0)
        ax.axvline(sigma_lo, color="0.4", ls="-.", lw=0.6, alpha=0.6, zorder=0)
        ax.axvline(sigma_hi, color="0.4", ls="-.", lw=0.6, alpha=0.6, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Noise level $\sigma$", labelpad=1)
        ax.set_ylabel("OOD RMSE", labelpad=2)
        ax.tick_params(axis="both", pad=2)
        ax.text(sigma_hi * 1.5, ax.get_ylim()[1] * 0.30,
                r"$\sigma_\star$", fontsize=7.5, color="0.3",
                va="top", ha="left")

        # Two-tier legend: d (colour + marker) and commitment (linestyle, grey).
        d_handles = [
            mlines.Line2D([], [], color=d_colors[d], marker=d_markers[d],
                          ms=3.5, lw=1.4, ls="-", label=f"$d={d}$")
            for d in ds
        ]
        cond_handles = [
            mlines.Line2D([], [], color="0.4", lw=1.2, ls=COND_LINESTYLE[c],
                          label=COND_LABEL[c])
            for c in CONDITIONS
        ]
        leg = ax.legend(handles=d_handles + cond_handles, loc="lower right",
                        frameon=False, handlelength=1.4, handletextpad=0.4,
                        borderpad=0.2, labelspacing=0.2, fontsize=8)
        for txt in leg.get_texts():
            txt.set_bbox(dict(facecolor="white", alpha=0.75,
                              edgecolor="none", pad=0.5))
        fig.tight_layout(pad=0.3)
    return fig


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


if __name__ == "__main__":
    fig = render()
    fig
