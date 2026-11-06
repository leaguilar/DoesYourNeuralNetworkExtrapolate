"""Exp 2.3 (CDS-frame) concept figure. Three panels:

  1. CDS-frame setup: schematic strand of yeast DNA with codon brackets,
     binary frame=0 target (1 every 3 bp).
  2. Ablation: length-OOD Pearson r per PE, on the random-init transformer.
     Codon P=3 wins; learned APE collapses; wrong / no PE near chance.
  3. Rashomon init sweep: r_OOD-len vs init period for learnable single-frequency
     PE. Sharp peaks at {1.5, 3, 6} (harmonics of 3).

Run as a script:
    conda run -n data_analysis python scripts/plot_concept.py --exp cds
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import (B_OOD, B_TRAIN, C_SIN, RC)

OUT = Path(__file__).resolve().parents[3] / "figures" / "exp2_3_frame_concept.pdf"
RESULTS_ROOT = Path(__file__).resolve().parents[3] / "results"

# Standard bioinformatics palette (per main/cds.py).  Gold-on-pale-yellow is
# illegible, so the G glyph itself uses DarkGoldenrod while its box keeps the
# pale-gold tint.
DNA_COLORS = {'A': '#228B22', 'T': '#CD5C5C', 'G': '#FFD700',
              'C': '#4682B4', 'N': '#808080'}


def _tint(hex_color, amount=0.85):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


DNA_TINTS = {b: _tint(c) for b, c in DNA_COLORS.items()}
DNA_LETTER_COLORS = dict(DNA_COLORS, G='#B8860B')


# %% data loaders
def _load_cds_full_summary(results_root: Path) -> dict | None:
    """Aggregate r_length-OOD by (backbone, PE) from exp2_3_full/."""
    d = results_root / "exp2_3_full"
    if not d.is_dir():
        return None
    rows = []
    for p in sorted(d.glob("*/summary.json")):
        with open(p) as f:
            data = json.load(f)
        rows.extend(data if isinstance(data, list) else [data])
    if not rows:
        return None
    agg: dict = {}
    for r in rows:
        key = (r["backbone"], r["pe_name"])
        agg.setdefault(key, []).append(r.get("r_ood_length"))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in agg.items()}


def _load_cds_init_sweep(results_root: Path) -> list | None:
    p = results_root / "exp2_3_learned_sweep" / "summary.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


# %% render — pure: builds and returns fig (or None when no data), no save, no close
def render():
    cds_summary = _load_cds_full_summary(RESULTS_ROOT)
    init_sweep = _load_cds_init_sweep(RESULTS_ROOT)

    if cds_summary is None and init_sweep is None:
        print("  [cds_frame] no results found, skipping")
        return None

    PE_BAR_ORDER = ["codon_pe", "multi_3_resmlp", "dual_pe", "learned_ape",
                    "nucleosome_pe", "standard_sinusoidal", "no_pe"]
    PE_BAR_LABEL = {
        "codon_pe":            r"Codon $P{=}3$",
        "multi_3_resmlp":      "Codon$+$res-MLP",
        "dual_pe":             "Codon$+$Helical",
        "learned_ape":         "Learned APE",
        "nucleosome_pe":       r"Helical $P{=}10.4$",
        "standard_sinusoidal": "Sinusoidal",
        "no_pe":               "No PE",
    }

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.0, 3.4))
        # Spacer columns let us widen the (a)|(b) gap while keeping (b)|(c) tight.
        gs = fig.add_gridspec(1, 5,
                              width_ratios=[1.05, 0.20, 1.10, 0.05, 1.20],
                              wspace=0.0)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[2])
        ax2 = fig.add_subplot(gs[4])

        for _ax, _lbl in zip((ax0, ax1, ax2), ("(a)", "(b)", "(c)")):
            _ax.text(-0.02, 1.22, _lbl, transform=_ax.transAxes,
                     fontsize=9, fontweight="bold", va="top", ha="left")

        # ── Panel 1: CDS-frame setup ─────────────────────────────────────
        ax = ax0
        ax.axis("off")
        seq = "ATGGCATTAGAACCAGTC"
        n = len(seq)
        CW = 1.0
        CH = 0.55
        Y_SEQ = 1.20
        Y_FRAME = 0.30
        Y_FORMULA = Y_FRAME + 0.25
        for i, base in enumerate(seq):
            r = mpatches.FancyBboxPatch(
                (i + 0.06, Y_SEQ - CH / 2), CW - 0.12, CH,
                boxstyle="round,pad=0.02",
                fc=DNA_TINTS.get(base, "#e8edf2"),
                ec="#444", lw=1.2, zorder=2)
            ax.add_patch(r)
            ax.text(i + CW / 2, Y_SEQ, base, ha="center", va="center",
                    fontsize=9, fontfamily="monospace",
                    color=DNA_LETTER_COLORS.get(base, "#222"),
                    fontweight="bold", zorder=3)

        for k in range(n // 3):
            xL = k * 3 + 0.06
            xR = (k + 1) * 3 - 0.06
            ax.annotate("", xy=(xR, Y_SEQ + CH / 2 + 0.10),
                        xytext=(xL, Y_SEQ + CH / 2 + 0.10),
                        arrowprops=dict(arrowstyle="|-|",
                                        color=B_TRAIN, lw=0.7,
                                        mutation_scale=2.5))

        # Y-range shrank from 3.2 → 2.5 to bring xt/yt closer; compensate the
        # circle's y-radius so the markers don't render as tall ovals.
        Y_RANGE_SCALE = 2.5 / 3.2
        for i in range(n):
            is_frame0 = (i % 3 == 0)
            colour = C_SIN if is_frame0 else "#ffffff"
            edge = B_TRAIN if is_frame0 else "#cccccc"
            r = mpatches.Ellipse(
                (i + CW / 2, Y_FRAME),
                width=0.44, height=0.44 * Y_RANGE_SCALE,
                fc=colour, ec=edge, lw=0.8, zorder=2)
            ax.add_patch(r)
            ax.text(i + CW / 2, Y_FRAME, "1" if is_frame0 else "0",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if is_frame0 else "#888",
                    fontweight="bold", zorder=3)
        ax.text(-0.5, Y_SEQ, r"$x_t$:", ha="right", va="center",
                fontsize=8, color="#666")
        ax.text(-0.5, Y_FRAME, r"$y_t$:", ha="right", va="center",
                fontsize=8, color=B_TRAIN, fontweight="bold")
        ax.text(n / 2, Y_FORMULA,
                r"$y_t = \mathbb{1}[t \,\mathrm{mod}\, 3 = 0]$",
                ha="center", va="bottom", fontsize=7.5, color=B_TRAIN)
        ax.set_xlim(-2.5, n + 0.5)
        ax.set_ylim(-0.1, 2.4)

        # ── Panel 2: ablation bar chart ──────────────────────────────────
        ax = ax1
        if cds_summary is not None:
            xs, means, stds, colours = [], [], [], []
            grey_ramp = ["#5a5a5a", "#7a7a7a", "#9a9a9a", "#bdbdbd"]
            grey_idx = 0
            for i, pe in enumerate(PE_BAR_ORDER):
                key = ("tiny_transformer", pe)
                if key not in cds_summary:
                    continue
                m, s = cds_summary[key]
                xs.append(PE_BAR_LABEL[pe])
                means.append(m)
                stds.append(s)
                if pe == "codon_pe":
                    colours.append(C_SIN)
                elif pe == "multi_3_resmlp":
                    colours.append(B_TRAIN)
                elif pe == "learned_ape":
                    colours.append(B_OOD)
                else:
                    colours.append(grey_ramp[min(grey_idx, len(grey_ramp) - 1)])
                    grey_idx += 1
            x_pos = np.arange(len(xs))
            ax.bar(x_pos, means, yerr=stds, color=colours, alpha=0.85,
                   capsize=3, lw=0.5, edgecolor="#444")
            ax.axhline(0.0, color="#aaa", lw=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xs, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(r"length-OOD $r$  (train $W{=}200$, eval $W{=}500$)",
                          fontsize=8)
            ax.set_ylim(-0.05, 1.0)
            ax.set_title("Right $\\varphi$ extrapolates; learned PE collapses",
                         fontsize=8.5, pad=14)
        else:
            ax.text(0.5, 0.5, "no exp2_3_full results",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#888")
            ax.set_xticks([]); ax.set_yticks([])

        # ── Panel 3: Rashomon init-period sweep ─────────────────────────
        ax = ax2
        if init_sweep is not None:
            by_init = defaultdict(list)
            for r in init_sweep:
                by_init[round(float(r["init_period_request"]), 3)].append(
                    r["r_ood_length"])
            xs = sorted(by_init.keys())
            means = [float(np.mean(by_init[k])) for k in xs]
            stds  = [float(np.std(by_init[k]))  for k in xs]

            ax.errorbar(xs, means, yerr=stds, fmt="o-", color=B_TRAIN,
                        ecolor="#888", capsize=2, lw=1.2, ms=4, zorder=3)
            for harm in [1.5, 3.0, 6.0]:
                ax.axvline(harm, color=C_SIN, lw=0.8, ls="--", alpha=0.6,
                           zorder=1)
            ax.axhline(0.13, color=B_OOD, lw=0.6, ls=":", alpha=0.7)
            ax.text(2.05, 0.16, "chance",
                    ha="left", va="bottom", fontsize=6.5, color=B_OOD,
                    style="italic")

            ax.set_xscale("log")
            ax.set_xticks([1.5, 3, 6, 10])
            ax.set_xticklabels(["1.5", "3", "6", "10"])
            ax.set_xticks([], minor=True)
            ax.set_xlim(1.3, 12.0)
            ax.set_xlabel(r"init period $P_0$ (log scale)", fontsize=8)
            ax.set_ylabel(r"length-OOD $r$", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("SGD does not move $\\log\\omega$; only $P_0\\!\\in\\!\\{1.5,3,6\\}$ "
                         "extrapolate",
                         fontsize=8.5, pad=14)

            for harm in [1.5, 3.0, 6.0]:
                if harm in by_init:
                    y = float(np.mean(by_init[harm]))
                    ax.annotate(f"$P\\!=\\!{harm:g}$",
                                xy=(harm, y),
                                xytext=(0, 8), textcoords="offset points",
                                ha="center", fontsize=6.5, color=C_SIN,
                                fontweight="bold")
        else:
            ax.text(0.5, 0.5, "no exp2_3_learned_sweep",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#888")
            ax.set_xticks([]); ax.set_yticks([])

        fig.tight_layout()
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
