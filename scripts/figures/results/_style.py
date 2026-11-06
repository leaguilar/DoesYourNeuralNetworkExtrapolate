"""Shared rcParams, condition palette, DGP markers, and torus colours."""

RC = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "lines.linewidth":   1.5,
}

# Colorblind-safe palette (per training condition)
C = {
    "raw_mlp":             "#e74c3c",
    "fourier_mlp":         "#2ecc71",
    "fourier_ols":         "#27ae60",
    "taylor9_direct":      "#c0392b",
    "raw":                 "#e74c3c",
    "fft":                 "#3498db",
    "wavelet":             "#9b59b6",
    "log_log_ols":         "#2ecc71",
    "log_y_ols":           "#3498db",
    "deriv":               "#27ae60",
    "poly2":               "#f39c12",
    "fourier":             "#2ecc71",
    "gating_fourier+poly2":"#8e44ad",
    "fno_baseline":        "#3498db",
    "siren_mlp":           "#f39c12",
    "log_ols":             "#3498db",
    "logp_ols":            "#8e44ad",
    "logp_mlp":            "#2ecc71",
    "default":             "#95a5a6",
}

LABELS = {
    "raw_mlp":             "Raw (MLP)",
    "fourier_mlp":         "Fourier (MLP)",
    "fourier_ols":         "Fourier (OLS)",
    "taylor9_direct":      "Taylor-9 (direct)",
    "raw":                 "Raw",
    "fft":                 "FFT (MLP)",
    "wavelet":             "Wavelet (MLP)",
    "log_log_ols":         r"Log-log (OLS)",
    "log_y_ols":           r"Log-$y$ (OLS)",
    "deriv":               "Derivative (OLS)",
    "poly2":               "Poly-2 (OLS)",
    "fourier":             "Fourier (MLP)",
    "gating_fourier+poly2":"Gating (MLP)",
    "poly7_ols":           "Poly-7 (OLS)",
    "poly9_ols":           "Poly-9 (OLS)",
    "fourier_ols_sel":     "Fourier (OLS)",
    "fno_baseline":        "FNO baseline",
    "siren_mlp":           "SIREN (MLP)",
    "log_ols":             r"Log (OLS)",
    "logp_ols":            r"Log-periodic (OLS)",
    "logp_mlp":            r"Log-periodic (MLP)",
}

DGP_COLORS = {
    "sin(x)":  "#2ecc71",
    "sin(2x)": "#3498db",
    "x^2":     "#f39c12",
    "x^3+x":   "#e74c3c",
}

DGP_MARKERS = {
    "sin(x)":  "o",
    "sin(2x)": "s",
    "x^2":     "^",
    "x^3+x":   "D",
}

# Torus benchmark (Nagler v2c) colours
TOR_C_RAW   = "#999999"
TOR_C_LEARN = "#e07a3c"
TOR_C_EXACT = "#2ecc71"


def col(cond):
    return C.get(cond, C["default"])


def lbl(cond):
    return LABELS.get(cond, cond.replace("_", " "))
