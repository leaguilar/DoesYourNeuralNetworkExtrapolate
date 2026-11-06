"""Shared parametrised builder used by the five Exp 3.3 (Arch-Inv) sweep tables (A/B/C/D/L)."""

import sys
from collections import defaultdict
from pathlib import Path

# Make _fmt importable as a top-level module when this file is loaded as a
# script (e.g. in VS Code 'Run Cell' mode where __package__ is not set).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _fmt import agg, fmt


MODEL_ORDER   = ["transformer", "mamba", "s4"]
MODEL_DISPLAY = {
    "transformer": r"Transformer",
    "mamba":       r"Mamba",
    "s4":          r"S4D",
}
PE_ORDER = ["exact_fourier",
            "learned_fourier_1freq", "learned_fourier_kfreq",
            "learnable_sinusoidal", "standard_sinusoidal",
            "rope", "learned_ape", "no_pe"]
PE_DISPLAY = {
    "exact_fourier":         r"\textbf{Exact Fourier}",
    "learned_fourier_1freq": r"Learned Fourier ($k{=}1$)",
    "learned_fourier_kfreq": r"Learned Fourier ($k{=}4$)",
    "learnable_sinusoidal":  r"Learnable Sinusoidal",
    "standard_sinusoidal":   r"Sinusoidal",
    "rope":                  r"RoPE",
    "learned_ape":           r"Learned APE",
    "no_pe":                 r"No PE",
}


def sweep_table(runs, axis_key, axis_label, axis_values,
                model_pe_pairs, caption, label,
                show_id=True, show_ood=True):
    """Generic builder: rows = (model, pe), columns = axis_values × {ID, OOD}.

    axis_key: which JSON field varies across columns (e.g. 'L_train', 'L_ood',
              'd_state_ssm').
    """
    by = defaultdict(list)
    for r in runs:
        v = r.get(axis_key)
        if v is None or v not in axis_values:
            continue
        by[(r["model_type"], r["pe_type"], v)].append(r)
    if not by:
        return ""

    n_metrics = (1 if show_id else 0) + (1 if show_ood else 0)
    col_spec = "ll" + "r" * (n_metrics * len(axis_values))
    head_top = (r"\textbf{Model} & \textbf{PE} & "
                + " & ".join(rf"\multicolumn{{{n_metrics}}}{{c}}"
                             rf"{{${axis_label}{{=}}{v}$}}"
                             for v in axis_values))
    metric_cells = []
    if show_id:  metric_cells.append("ID")
    if show_ood: metric_cells.append("OOD")
    head_sub = " & & " + " & ".join(" & ".join(metric_cells)
                                     for _ in axis_values)
    cmid = " ".join(
        rf"\cmidrule(lr){{{2 + n_metrics*i + 1}-{2 + n_metrics*(i+1)}}}"
        for i in range(len(axis_values)))

    wide = (n_metrics * len(axis_values) >= 6)

    out = (f"\\begin{{table}}[ht]\n  \\centering\n  \\small\n"
           f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n")
    if wide:
        out += "  \\resizebox{\\textwidth}{!}{%\n"
    out += (f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n"
            f"    {head_top} \\\\\n    {cmid}\n"
            f"    {head_sub} \\\\\n    \\midrule\n")

    for model, pe in model_pe_pairs:
        if not any(by.get((model, pe, v)) for v in axis_values):
            continue
        cells = []
        for v in axis_values:
            rs = by.get((model, pe, v))
            if not rs:
                cells.extend(["---"] * n_metrics)
                continue
            if show_id:
                m, s = agg([r["err_id"] * 100 for r in rs])
                cells.append(fmt(m, s, decimals=1))
            if show_ood:
                m, s = agg([r["err_ood"] * 100 for r in rs])
                cells.append(fmt(m, s, decimals=1))
        shade = "    \\rowcolor{gray!12}\n" if pe == "exact_fourier" else ""
        out += (f"{shade}    {MODEL_DISPLAY[model]} & "
                f"{PE_DISPLAY[pe]} & "
                + " & ".join(cells) + " \\\\\n")
    out += "    \\bottomrule\n  \\end{tabular}\n"
    if wide:
        out += "  }\n"
    out += "\\end{table}\n"
    return out
