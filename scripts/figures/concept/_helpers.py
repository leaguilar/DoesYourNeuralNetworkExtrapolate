"""Shared helpers for concept figures: zone shading, Taylor expansion."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches

# Make _style importable as a top-level module when this file is loaded as a
# script (e.g. in VS Code 'Run Cell' mode where __package__ is not set).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import C_TRAIN, C_NEAR, C_OOD, B_TRAIN, B_NEAR, B_OOD


def shade_zone(ax, x0, x1, color, border, label, y0=-1e9, y1=1e9, zorder=0):
    ax.axvspan(x0, x1, ymin=0, ymax=1, alpha=0.18, color=color, zorder=zorder)
    ax.axvline(x0, color=border, lw=0.7, ls=":", zorder=zorder + 1)
    ax.axvline(x1, color=border, lw=0.7, ls=":", zorder=zorder + 1)


def zone_legend_handles():
    return [
        mpatches.Patch(fc=C_TRAIN, ec=B_TRAIN, lw=0.8, label="Training"),
        mpatches.Patch(fc=C_NEAR,  ec=B_NEAR,  lw=0.8, label="Near-boundary val."),
        mpatches.Patch(fc=C_OOD,   ec=B_OOD,   lw=0.8, label="OOD evaluation"),
    ]


def taylor9(x):
    """sin(x) Taylor series around 0, degree 9."""
    return x - x**3 / 6 + x**5 / 120 - x**7 / 5040 + x**9 / 362880
