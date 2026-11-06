#!/usr/bin/env python3
"""Exp 4.4 companion: squared-OOD-error ratio raw : exact-φ per (σ, d).

Answers the *relative* question that OOD R^2 stops answering once every arm is
negative: how much does committing to the canonical φ buy over committing to
nothing? The statistic is the per-seed ratio

    (OOD RMSE of raw_mlp / OOD RMSE of exact_phi_mlp)^2,

i.e. the factor by which the raw arm's squared OOD error exceeds the canonical
arm's, aggregated as mean ± np.std over seeds (same `agg()` convention as every
other table in the paper).

The point of the table: at σ = 0.01 the commitment buys ~1{,}950x at d = 4, and
as σ grows the ratio compresses onto a floor near 2 and stops there — the same
floor at d = 6 and d = 8. Once label noise dominates, both arms pay the same
irreducible noise term and the commitment's advantage saturates at a constant
factor rather than vanishing.

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _loaders import load_results
from _fmt import agg, fmt, table_footer, table_header

EID = "4.4"
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp4_4_ratio.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"

NUM = "raw_mlp"          # numerator arm: no commitment
DEN = "exact_phi_mlp"    # denominator arm: canonical commitment
SIGMAS = (0.3, 1.0, 3.0, 10.0, 30.0)


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp4_4_ratio] no input found, skipping")
        return None

    # by[d][sigma] -> list of per-seed squared-error ratios
    by = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for cell in run["conditions"].values():
            d = int(cell["d"])
            sigma = float(cell["sigma"])
            num = cell[NUM]["ood_rmse"]
            den = cell[DEN]["ood_rmse"]
            if den <= 0:
                continue
            by[d][sigma].append((num / den) ** 2)

    ds = sorted(by.keys())
    if not ds:
        print(f"  [table_exp4_4_ratio] no usable cells, skipping")
        return None
    n_seeds = len(runs)

    # Anchor value quoted in the caption: the low-noise ratio at the smallest d.
    d0 = ds[0]
    anchor_sigma = 0.01
    anchor = (float(np.mean(by[d0][anchor_sigma]))
              if anchor_sigma in by[d0] else float("nan"))

    out = []
    out.append(table_header(
        caption=(
            r"Exp.~4.4, relative view: per-seed squared OOD error of the "
            r"uncommitted arm over the canonical arm, "
            r"$(\mathrm{RMSE}^{\mathrm{ood}}_{\mathrm{raw}} / "
            r"\mathrm{RMSE}^{\mathrm{ood}}_{\text{exact-}\varphi})^2$, "
            r"mean$\pm$std over " + f"{n_seeds}" + r" seeds. "
            r"Where OOD $R^2$ (\Cref{tab:exp44}) answers the absolute question "
            r"\emph{does this arm beat predicting the mean} and is negative "
            r"for every arm once $\sigma$ is large, this ratio answers the "
            r"complementary relative question \emph{how much does the "
            r"commitment buy over no commitment}, and stays meaningful after "
            r"every $R^2$ has gone negative. Reading down the "
            + rf"$d{{=}}{d0}$ column, the advantage compresses from "
            + (f"{anchor:,.0f}".replace(",", "{,}") if np.isfinite(anchor)
               else r"\textit{n/a}")
            + rf"$\times$ at $\sigma{{=}}{anchor_sigma:g}$ (off-table) onto a "
            r"floor near $2\times$ and then stops; the same floor is reached "
            r"at the higher $d$, only at larger $\sigma$. Once label noise "
            r"dominates, both arms pay the same irreducible noise term, so "
            r"the commitment's advantage saturates at a constant factor "
            r"rather than vanishing."
        ),
        label="tab:exp44_ratio",
        col_spec="l" + "r" * len(ds),
        header_row=(r"$\sigma$ & "
                    + " & ".join(rf"$d = {d}$" for d in ds)),
    ))

    for sigma in SIGMAS:
        cells = []
        for d in ds:
            vals = by[d].get(sigma, [])
            if vals:
                m, s = agg(vals)
                cells.append(fmt(m, s, bold=False, decimals=1))
            else:
                cells.append("---")
        out.append(f"    {sigma:g} & " + " & ".join(cells) + r" \\")

    out.append(table_footer())
    return "\n".join(out) + "\n"


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
    tex = render()
    if tex:
        print(tex)
