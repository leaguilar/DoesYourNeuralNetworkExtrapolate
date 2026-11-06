"""Main-paper CDS concept: 18-bp DNA strand with three universal-genetic-code
hypotheses overlaid (codon period P=3, no in-frame stops, biological AA).

Run as a script:
    conda run -n data_analysis python scripts/main_plot_concept.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import B_TRAIN, RC

OUT = Path(__file__).resolve().parents[3] / "figures" / "main" / "exp2_3_species_concept_main.pdf"

DNA_COLORS = {
    'A': '#228B22',  # Forest Green
    'T': '#CD5C5C',  # Indian Red
    'G': '#FFD700',  # Gold / Yellow
    'C': '#4682B4',  # Steel Blue
    'N': '#808080',  # Grey for unknown/null
}


def _tint(hex_color, amount=0.85):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


DNA_TINTS = {b: _tint(c) for b, c in DNA_COLORS.items()}

# Gold #FFD700 is too light to read on pale yellow — render the G glyph in
# DarkGoldenrod so it remains in the yellow family but contrasts with its tint.
DNA_LETTER_COLORS = dict(DNA_COLORS, G='#B8860B')


# %% render — pure: builds and returns fig, no save, no close
def render():
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.0))

        ax.axis("off")
        seq = "ATGGCATTAGAACCAGTC"
        n = len(seq)
        CW, Y_SEQ, CH = 1.0, 2.00, 0.55
        for i, base in enumerate(seq):
            r = mpatches.FancyBboxPatch(
                (i + 0.06, Y_SEQ - CH / 2), CW - 0.12, CH,
                boxstyle="round,pad=0.02",
                fc=DNA_TINTS.get(base, "#e8edf2"),
                ec="#444", lw=1.2, zorder=2)
            ax.add_patch(r)
            ax.text(i + CW / 2, Y_SEQ, base, ha="center", va="center",
                    fontsize=12.75, fontfamily="monospace",
                    color=DNA_LETTER_COLORS.get(base, "#222"),
                    fontweight="bold", zorder=3)
        for k in range(n // 3):
            xL = k * 3 + 0.06
            xR = (k + 1) * 3 - 0.06
            ax.annotate("", xy=(xR, Y_SEQ + CH / 2 + 0.10),
                        xytext=(xL, Y_SEQ + CH / 2 + 0.10),
                        arrowprops=dict(arrowstyle="|-|",
                                        color="black", lw=0.7,
                                        mutation_scale=2.5))
        ax.text(n / 2, Y_SEQ + CH / 2 + 0.28, "codon period $P{=}3$",
                ha="center", va="bottom", fontsize=9.5, color="black",
                fontstyle="italic")
        bullets = [
            (r"$\mathbf{H_1}$: codon period $P{=}3$  $\to$  $|\mathrm{FFT}|@P{=}3$", 1.20),
            (r"$\mathbf{H_2}$: zero stops in true frame  $\to$  count($\mathtt{TAA}$, $\mathtt{TAG}$, $\mathtt{TGA}$)", 0.75),
            (r"$\mathbf{H_3}$: protein-typical AA  $\to$  per-frame 21-class histogram", 0.30),
        ]
        for txt, y in bullets:
            ax.text(0.0, y, txt, ha="left", va="center",
                    fontsize=12, color="#333")
        ax.text(-0.5, Y_SEQ, r"$x_t$:", ha="right", va="center",
                fontsize=12, color="#666")
        ax.set_xlim(-2.5, n + 0.5)
        ax.set_ylim(0.05, 2.95)

        fig.tight_layout()
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
