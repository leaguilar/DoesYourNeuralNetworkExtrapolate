"""Shared LaTeX formatting helpers and the condition-name label dictionary."""

import math

import numpy as np


def _latex_thousands(n):
    """Format an integer with LaTeX thin-space thousand separator (e.g., 1881 -> '1{,}881')."""
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return str(n)
    return f"{n_int:,}".replace(",", "{,}")


def agg(vals):
    return float(np.mean(vals)), float(np.std(vals))


CONDITION_LABELS = {
    "raw_mlp":             r"\texttt{raw\_mlp} (MLP)",
    "fourier_mlp":         r"\texttt{fourier\_mlp} (MLP)",
    "fourier_ols":         r"\texttt{fourier\_ols} (OLS)",
    "taylor9_direct":      r"\texttt{taylor9} (direct)",
    "timesfm":             r"\texttt{TimesFM} (zero-shot)",
    "raw":                 r"\texttt{raw}",
    "fft":                 r"\texttt{fft} (MLP)",
    "wavelet":             r"\texttt{wavelet} (MLP)",
    "log_log_ols":         r"\texttt{log-log} (OLS)",
    "log_y_ols":           r"\texttt{log-}$y$ (OLS)",
    "log_log_mlp":         r"\texttt{log-log} (MLP)",
    "log_y_mlp":           r"\texttt{log-}$y$ (MLP)",
    "deriv":               r"\texttt{deriv} (OLS)",
    "deriv_ols":           r"\texttt{deriv} (OLS)",
    "poly2":               r"\texttt{poly2}",
    "fourier":             r"\texttt{fourier} (MLP)",
    "gating_fourier+poly2":r"\texttt{gating} (MLP)",
    "fourier_ols_sel":     r"\texttt{fourier\_ols}",
    "poly7_ols":           r"\texttt{poly7\_ols}",
    "poly9_ols":           r"\texttt{poly9\_ols}",
    "raw_ols":             r"\texttt{raw\_ols}",
    "log_ols":             r"\texttt{log\_ols} (OLS)",
    "logp_ols":            r"\texttt{logp\_ols} (OLS)",
    "logp_mlp":            r"\texttt{logp\_mlp} (MLP)",
    "fno_baseline":        r"\texttt{fno\_baseline}",
    "siren_mlp":           r"\texttt{siren\_mlp}",
    "log_log_poly2":       r"\texttt{log-log} (OLS deg-2)",
    "log_y_poly2":         r"\texttt{log-}$y$ (OLS deg-2)",
    "raw_sindy":           r"\texttt{raw\_sindy} (SINDy)",
    "log_log_sindy":       r"\texttt{log-log} (SINDy)",
    "log_y_sindy":         r"\texttt{log-}$y$ (SINDy)",
    "log_log_node":        r"\texttt{log-log} (NODE)",
    "log_y_node":          r"\texttt{log-}$y$ (NODE)",
    "tabpfn_raw":          r"\texttt{tabpfn\_raw} (TabPFN)",
    "tabpfn_fourier":      r"\texttt{tabpfn\_fourier} (TabPFN)",
    "tabpfn_loglog":       r"\texttt{tabpfn\_loglog} (TabPFN)",
    "tabpfn_logy":         r"\texttt{tabpfn\_logy} (TabPFN)",
    "timesfm_raw":         r"\texttt{timesfm\_raw} (TimesFM)",
    "timesfm_loglog":      r"\texttt{timesfm\_loglog} (TimesFM)",
    "timesfm_logy":        r"\texttt{timesfm\_logy} (TimesFM)",
}


def lbl(c):
    return CONDITION_LABELS.get(c, r"\texttt{" + c.replace("_", r"\_") + "}")


def _fmt_with_thousands(value, decimals=1):
    """Format `value` with `decimals` digits and a LaTeX-friendly thousands
    separator (`{,}` keeps the spacing tight in math mode)."""
    if abs(value) < 1000:
        return f"{value:.{decimals}f}"
    return f"{value:,.{decimals}f}".replace(",", "{,}")


def fmt(mean, std, bold=False, decimals=1):
    if math.isnan(mean):
        s = r"\textit{Div.}"
    elif mean > 9999:
        s = r"$\gg 9{,}999$"
    else:
        s = (f"{_fmt_with_thousands(mean, decimals)} $\\pm$ "
             f"{_fmt_with_thousands(std, decimals)}")
    return f"\\textbf{{{s}}}" if bold else s


def best_bold(vals_dict, metric="ood_error"):
    best = min(v[metric]["mean"] for v in vals_dict.values())
    return best


def table_header(caption, label, col_spec, header_row):
    return (
        f"\\begin{{table}}[ht]\n"
        f"  \\centering\n"
        f"  \\small\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        f"    \\toprule\n"
        f"    {header_row} \\\\\n"
        f"    \\midrule\n"
    )


def table_footer():
    return "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
