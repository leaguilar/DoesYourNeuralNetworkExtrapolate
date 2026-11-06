#!/usr/bin/env python3
"""Exp 2.4 (Camelyon17): cross-hospital tumour classification under a fixed
input feature map phi (Macenko stain normalisation).

ResNet-18 trained FROM SCRATCH on centres {0,3,4}, held-out OOD centre 2,
5 seeds, numbers taken at the ID-selected epoch (model selection never touches
OOD labels). phi turns a 0.75 OOD-AUC collapse into 0.98.

The frozen pretrained-backbone + linear-probe arm is NOT reported: those
backbones have unknown pretraining exposure (they may already have seen
histopathology), so the comparison is confounded, and it was a single seed.

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import json
import math
import sys
from pathlib import Path

import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _loaders import load_histopath
from _fmt import (
    CONDITION_LABELS,
    _fmt_with_thousands,
    _latex_thousands,
    agg,
    best_bold,
    fmt,
    lbl,
    table_footer,
    table_header,
)

OUT = Path(__file__).resolve().parents[2] / "tables" / "table_histopath.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"

# Arm order and display labels, shared with the main-paper excerpt.
ARM_ORDER = ["raw", "stainnorm", "channorm", "colorctrl", "raw+caug"]
ARM_LABELS = {
    "raw":       r"\texttt{raw}",
    "stainnorm": r"\texttt{stainnorm} ($\varphi$)",
    "channorm":  r"\texttt{channorm}",
    "colorctrl": r"\texttt{colorctrl}",
    "raw+caug":  r"\texttt{raw+caug}",
}
# Rows that actually apply phi get the gray shade.
PHI_ARMS = {"stainnorm", "channorm"}

# Cross-hospital split, frozen in src/data/camelyon_cache/manifest.json.
TRAIN_CENTRES = [0, 3, 4]
OOD_CENTRE = 2

SHADE = r"\rowcolor{gray!12}"
N_COLS = 5


def _load_input(results_dir=None):
    return load_histopath(str(results_dir or RESULTS_DIR_DEFAULT))


def _resizebox(tex):
    """Wrap the tabular emitted by table_header/table_footer in \\resizebox.

    Four `mean $\\pm$ std` columns at 4 decimals run ~17pt past the NeurIPS
    text width; same treatment as the wide Exp 3.3 arch-invariance grid.
    """
    tex = tex.replace("  \\begin{tabular}",
                      "  \\resizebox{\\linewidth}{!}{%\n  \\begin{tabular}", 1)
    tex = tex.replace("  \\end{tabular}\n\\end{table}",
                      "  \\end{tabular}%\n  }\n\\end{table}", 1)
    return tex


def _scratch_stats(runs):
    """{arm: {metric: (mean, std)}} over seeds, at the ID-selected epoch."""
    stats = {}
    for arm in ARM_ORDER:
        rows = [r for r in runs if r.get("arm_key") == arm]
        if not rows:
            continue
        cell = {}
        for metric, (split, key) in {
            "id_acc":  ("id",  "acc"),
            "id_auc":  ("id",  "auc"),
            "ood_acc": ("ood", "acc"),
            "ood_auc": ("ood", "auc"),
        }.items():
            cell[metric] = agg([r["id_selected"][split][key] for r in rows])
        cell["n_seeds"] = len(rows)
        stats[arm] = cell
    return stats


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs or not runs.get("scratch"):
        print("  [table_histopath] no input found, skipping")
        return None

    scratch = _scratch_stats(runs["scratch"])

    n_seeds = max(c["n_seeds"] for c in scratch.values())
    centres_str = r"\{" + ",\\,".join(str(c) for c in TRAIN_CENTRES) + r"\}"
    ood_str = str(OOD_CENTRE)

    r0 = runs["scratch"][0]
    n_train = r0.get("n_train")
    n_id = r0.get("n_idtest")
    n_ood = r0.get("n_oodtest")

    cap = (
        r"Exp.~2.4 (Camelyon17): cross-hospital tumour classification under a "
        r"fixed input feature map $\varphi$. "
        r"Camelyon17-WILDS $96{\times}96$ H\&E lymph-node patches "
        r"(tumour vs.\ normal); training centres "
        rf"${centres_str}$, held-out OOD centre ${ood_str}$. "
        r"$\varphi$ is Macenko stain normalisation, derived from the "
        r"Beer--Lambert optical-density model of dye absorption and fit "
        r"\emph{once} on a single training patch: it sees no OOD image and no "
        r"OOD statistic. "
        r"Arms: \texttt{raw}, no preprocessing; "
        r"\texttt{stainnorm}, the map $\varphi$; "
        r"\texttt{channorm}, per-image per-channel mean/std standardisation "
        r"(adaptive but stain-agnostic, it never aligns the stain basis); "
        r"\texttt{colorctrl}, one fixed random RGB affine applied identically "
        r"to every image and every hospital (negative control, matched in form "
        r"to $\varphi$ but non-adaptive); "
        r"\texttt{raw+caug}, training-time stochastic colour jitter (the "
        r"standard domain-generalisation baseline). "
        rf"ResNet-18 trained from scratch "
        rf"($n_\mathrm{{train}}={_latex_thousands(n_train)}$, "
        rf"$n_\mathrm{{ID}}={_latex_thousands(n_id)}$, "
        rf"$n_\mathrm{{OOD}}={_latex_thousands(n_ood)}$), reported at the "
        r"\emph{ID-selected} epoch (model selection uses in-distribution "
        r"validation only and never sees OOD labels), mean $\pm$ std over "
        rf"{n_seeds} seeds. "
        r"\colorbox{gray!12}{\strut Shaded rows} are input maps $\varphi$; best OOD AUC "
        r"in bold. "
        r"ID AUC is $\ge 0.995$ for every arm, so the entire spread is "
        r"out-of-distribution: $\varphi$ converts an OOD-AUC collapse to "
        r"$0.75$ into $0.98$, while the non-adaptive \texttt{colorctrl} "
        r"control deepens it to $0.63$."
    )

    out = table_header(
        cap, "tab:histopath", "lrrrr",
        r"\textbf{Input map} & \textbf{ID acc} & \textbf{ID AUC} & "
        r"\textbf{OOD acc} & \textbf{OOD AUC}"
    )

    # ── Block 1: ResNet-18 from scratch (5 seeds, mean ± std) ────────────────
    out += (f"    \\multicolumn{{{N_COLS}}}{{l}}{{\\emph{{ResNet-18, trained "
            f"from scratch ({n_seeds} seeds)}}}} \\\\\n")
    out += "    \\midrule\n"

    best_scratch = max(c["ood_auc"][0] for c in scratch.values())
    for arm in ARM_ORDER:
        if arm not in scratch:
            continue
        c = scratch[arm]
        is_best = abs(c["ood_auc"][0] - best_scratch) < 1e-9
        prefix = f"    {SHADE}\n    " if arm in PHI_ARMS else "    "
        out += (f"{prefix}{ARM_LABELS[arm]} & "
                f"{fmt(*c['id_acc'],  decimals=4)} & "
                f"{fmt(*c['id_auc'],  decimals=4)} & "
                f"{fmt(*c['ood_acc'], decimals=4)} & "
                f"{fmt(*c['ood_auc'], bold=is_best, decimals=4)} \\\\\n")

    # Frozen pretrained backbones are deliberately not reported: those weights
    # have unknown pretraining exposure (possibly including histopathology), so
    # that arm is confounded rather than a clean comparison, and it is 1 seed.

    out += table_footer()
    return _resizebox(out)


# %% build — CLI / shell entry point
def build(runs=None, output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = render(runs)
    if tex is None:
        return None
    out.write_text(tex)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    print(render())
