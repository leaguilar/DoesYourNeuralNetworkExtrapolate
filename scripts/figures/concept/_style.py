"""Shared rcParams and zone/DGP colour palette for concept figures."""

RC = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.2,
    "lines.linewidth":   1.6,
}

# Zone fills (light) and borders / line colours
C_TRAIN  = "#d5e8f5"
C_NEAR   = "#fde8cc"
C_OOD    = "#fad5d5"

B_TRAIN  = "#5a8fa8"
B_NEAR   = "#c87137"
B_OOD    = "#b83232"

# DGP / curve colours
C_SIN    = "#2ecc71"
C_POLY   = "#2e3812"
C_EXP    = "#3498db"
