"""Empirical periodogram of yeast plus-strand AT-content, inside vs outside CDS.

Renders the FFT result that empirically justifies the codon-only oracle:
only P=3 bp is CDS-specific (the only period whose spectral mass is enriched
inside CDS), so adding helical or nucleosome periods to the PE injects no
CDS-discriminating signal.

Run as a script:
    conda run -n data_analysis python scripts/plot_concept.py --exp cds_fft
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import (B_OOD, B_TRAIN, C_SIN, RC)

OUT = Path(__file__).resolve().parents[3] / "figures" / "exp2_3_fft_periodogram.pdf"
RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results"


# %% render — pure: builds and returns fig (or None when no data), no save, no close
def render():
    spec_path = RESULTS_ROOT / "exp2_3_fft" / "spectrum.npz"
    if not spec_path.is_file():
        print("  [exp2_3_fft] no spectrum.npz, skipping")
        return None
    z = np.load(spec_path)
    freqs   = z["freqs"]
    periods = z["periods"]
    S_cds   = z["S_cds_at"]
    S_non   = z["S_non_at"]

    mask = freqs > 0.005
    f = freqs[mask]
    p = periods[mask]
    S_c = S_cds[mask]
    S_n = S_non[mask]
    ratio = S_c / np.maximum(S_n, 1e-9)

    with plt.rc_context(RC):
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(11.0, 3.2), sharex=False,
            gridspec_kw=dict(width_ratios=[1.2, 1.0], wspace=0.32))

        for _ax, _lbl in zip((ax0, ax1), ("(a)", "(b)")):
            _ax.text(0.0, 1.10, _lbl, transform=_ax.transAxes,
                     fontsize=9, fontweight="bold", va="top", ha="left")

        # Panel A: spectrum, inside vs outside CDS
        ax0.plot(p, S_c, color=B_TRAIN, lw=1.2, label="inside CDS")
        ax0.plot(p, S_n, color="#888888", lw=1.0, label="outside CDS")
        ax0.set_xscale("log")
        ax0.set_xlim(p.min(), p.max())
        ax0.axvline(3.0,  color=C_SIN, lw=1.0, ls="--", alpha=0.7)
        ax0.axvline(10.4, color=B_OOD, lw=1.0, ls="--", alpha=0.5)
        ax0.text(3.0, ax0.get_ylim()[1] * 0.95, "$P{=}3$ codon",
                 ha="center", va="top", fontsize=8, color=C_SIN,
                 fontweight="bold")
        ax0.text(10.4, ax0.get_ylim()[1] * 0.95, "$P{=}10.4$ helical",
                 ha="center", va="top", fontsize=8, color=B_OOD)
        ax0.set_xlabel("period (bp, log scale)", fontsize=8.5)
        ax0.set_ylabel("mean periodogram (AT indicator)", fontsize=8.5)
        ax0.set_title(r"Average power spectrum, $W{=}200$ windows",
                      fontsize=9, pad=6)
        ax0.legend(loc="upper right", fontsize=8, frameon=False)

        # Panel B: CDS / non-CDS ratio (the discriminative quantity)
        ax1.plot(p, ratio, color=B_TRAIN, lw=1.2)
        ax1.axhline(1.0, color="#aaa", lw=0.6, ls=":")
        ax1.axvline(3.0, color=C_SIN, lw=1.0, ls="--", alpha=0.7)
        ax1.axvline(10.4, color=B_OOD, lw=1.0, ls="--", alpha=0.5)
        ax1.set_xscale("log")
        ax1.set_xlim(p.min(), p.max())
        ax1.set_xlabel("period (bp, log scale)", fontsize=8.5)
        ax1.set_ylabel(r"CDS power / non-CDS power", fontsize=8.5)
        ax1.set_title(r"Only $P{=}3$ is CDS-specific", fontsize=9, pad=6)
        idx3 = int(np.argmin(np.abs(p - 3.0)))
        ax1.annotate(f"$P{{=}}3$\n ratio $\\approx${ratio[idx3]:.2f}",
                     xy=(p[idx3], ratio[idx3]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=7, color=C_SIN, fontweight="bold")
    return fig


# %% build — CLI / shell entry point
def build(output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = render()
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
