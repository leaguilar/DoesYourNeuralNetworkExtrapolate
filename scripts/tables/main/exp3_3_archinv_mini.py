#!/usr/bin/env python3
"""Exp 3.3 Architectural invariance (excerpt): 3 PE rows x 3 archs x 3 periods, compact layout.

Promotes the headline architecture-invariance contrast (Exact Fourier vs
canonical Sinusoidal vs unconstrained Learned APE) into the main text.
Architecture is a column (not a block header) and the chance row is dropped
to save vertical space; periods are subsampled to (smallest, medium, largest)
of the baseline set {64, 128, 256, 512}. The full 8-PE x 4-period grid stays
in the appendix (`tab:exp33_main`).

Run as a script:
    conda run -n data_analysis python scripts/main_make_tables.py
"""

# %% imports
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make sibling + parent modules importable in both shell-import and cell-mode
# contexts. The parent (scripts/tables) hosts _loaders, _fmt, etc.
_HERE = Path(__file__).resolve().parent
_TABLES = _HERE.parent
for _p in (_HERE, _TABLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _fmt import agg, fmt
from _loaders import load_exp3_3_results
from _exp3_3_archinv import MODEL_DISPLAY, MODEL_ORDER, PE_DISPLAY

OUT = Path(__file__).resolve().parents[3] / "tables" / "main" / "table_exp3_3_minitext.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[3] / "results"

PE_SUBSET = ("exact_fourier", "standard_sinusoidal", "learned_ape")
P_VALUES = [64, 256, 512]  # smallest / medium / largest of {64,128,256,512}


def _is_baseline(r):
    """Mirror the baseline filter from `make_tables.table_exp3_3`."""
    P = r["period"]
    if r.get("L_train") not in (None, 4 * P):
        return False
    if r.get("L_ood") not in (None, 2 * P):
        return False
    ds = r.get("d_state_ssm", 64)
    if ds is not None and ds != 64:
        return False
    return True


def _load_input(results_dir=None):
    return load_exp3_3_results(str(results_dir or RESULTS_DIR_DEFAULT))


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp3_3_minitext] no input found, skipping")
        return None

    runs = [r for r in runs
            if _is_baseline(r)
            and r.get("pe_type") in PE_SUBSET
            and r.get("period") in P_VALUES]
    if not runs:
        return ""

    # Dedupe (model, pe, P, seed) — same as the legacy builder, since a
    # cell may have a pure-baseline file plus a sweep file with coincident
    # hyperparameters.
    by_seed = {}
    for r in runs:
        key = (r["model_type"], r["pe_type"], r["period"], r["seed"])
        by_seed.setdefault(key, r)
    runs = list(by_seed.values())

    n_seeds = max(
        sum(1 for r in runs
            if (r["model_type"], r["pe_type"], r["period"]) == k)
        for k in {(r["model_type"], r["pe_type"], r["period"]) for r in runs}
    )
    n_total = len(runs)

    chance_err = (1.0 - 1.0 / runs[0]["vocab_size"]) * 100
    cap = (
        r"Exp.~3.3 Architectural invariance (excerpt). ID/OOD error (\%) on the "
        r"periodic annotation task across Transformer, Mamba, S4D and "
        r"$P \in \{" + ", ".join(str(p) for p in P_VALUES) + r"\}$ "
        r"($L_\mathrm{train}{=}4P$, $L_\mathrm{OOD}{=}2P$; chance "
        rf"${{\approx}}{chance_err:.2f}\%$). "
        r"\colorbox{gray!12}{\strut Shaded rows} use the correct PE; "
        r"the OOD guarantee belongs to the "
        r"feature map, not the backbone. Best OOD per (architecture, $P$) "
        r"in bold. "
        rf"Mean over {n_seeds} seeds; {n_total} runs. "
        r"Full $8$-PE $\times$ $4$-period grid in \Cref{tab:exp33_main}."
    )

    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["model_type"], r["pe_type"], r["period"])].append(r)

    # Lowest OOD per (model, P) within the kept PE rows -> drives bolding.
    best = {}
    for m in MODEL_ORDER:
        for P in P_VALUES:
            vals = []
            for pe in PE_SUBSET:
                rs = by_key.get((m, pe, P))
                if rs:
                    vals.append(float(np.mean(
                        [r["err_ood"] for r in rs])) * 100)
            best[(m, P)] = min(vals) if vals else None

    n_P = len(P_VALUES)
    col_spec = "ll" + "rr" * n_P
    head_top = (r"\textbf{Architecture} & \textbf{PE} & "
                + " & ".join(rf"\multicolumn{{2}}{{c}}{{$P{{=}}{P}$}}"
                             for P in P_VALUES))
    cmid = " ".join(rf"\cmidrule(lr){{{2*i+3}-{2*i+4}}}"
                    for i in range(n_P))
    head_sub = " & & " + " & ".join("ID & OOD" for _ in P_VALUES)

    out = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{{cap}}}",
        rf"  \label{{tab:exp33_minitext}}",
        r"  \resizebox{\textwidth}{!}{%",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    {head_top} \\\\",
        f"    {cmid}",
        f"    {head_sub} \\\\",
        r"    \midrule",
    ]

    for m in MODEL_ORDER:
        for i, pe in enumerate(PE_SUBSET):
            cells = []
            for P in P_VALUES:
                rs = by_key.get((m, pe, P)) or []
                if not rs:
                    cells.extend(["---", "---"])
                    continue
                id_m, id_s = agg([r["err_id"] * 100 for r in rs])
                oo_m, oo_s = agg([r["err_ood"] * 100 for r in rs])
                bold_ood = (best[(m, P)] is not None
                            and abs(oo_m - best[(m, P)]) < 1e-3)
                cells.append(fmt(id_m, id_s, decimals=1))
                cells.append(fmt(oo_m, oo_s, bold=bold_ood, decimals=1))
            shade = "    \\rowcolor{gray!12}\n" if pe == "exact_fourier" else ""
            model_cell = MODEL_DISPLAY[m] if i == 0 else ""
            out.append(f"{shade}    {model_cell} & {PE_DISPLAY[pe]} & "
                       + " & ".join(cells) + r" \\")

    out += [r"    \bottomrule", r"  \end{tabular}", r"  }", r"\end{table}", ""]
    return "\n".join(out)


# %% build — CLI / shell entry point
def build(runs=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = render(runs)
    if not tex:
        return None
    out.write_text(tex)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    tex = render()
    if tex:
        print(tex)
