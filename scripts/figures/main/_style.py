"""Shared rcParams and zone/DGP palette for main-paper concept figures.

Mirrors figures/concept/_style.py for the abbreviated main-paper variants.
Kept as a sibling to figures/concept/ so the main-paper styling can diverge
without touching the appendix figures.
"""

RC = {
    "font.family":       "serif",
    "font.size":         13.5,
    "axes.labelsize":    13.5,
    "axes.titlesize":    13.5,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   10.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "lines.linewidth":   1.6,
}

# Zone fills + borders
C_TRAIN  = "#d5e8f5"
C_OOD    = "#fad5d5"
B_TRAIN  = "#5a8fa8"
B_OOD    = "#b83232"

# DGP / curve colours
C_SIN    = "#2ecc71"
C_POLY   = "#e74c3c"
