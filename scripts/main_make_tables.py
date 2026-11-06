#!/usr/bin/env python3
"""Generate excerpt tables for the main-paper sections — dispatch shell.

Active main-paper artefacts live in `scripts/tables/main/*.py` (one file
per artefact) and are wired through DISPATCH below. The legacy
table_exp1_1_main and table_exp1_3_sindy_main builders are kept in this file
for ad-hoc invocation but are not run by `python scripts/main_make_tables.py`
because §sec:exp11 inputs the FULL table (per AUDIT D-001) and the SINDy
diagnostic surfaces in `table_sindy_delta_consolidated.tex`.

Usage (from project root):
    conda run -n data_analysis python scripts/main_make_tables.py
    conda run -n data_analysis python scripts/main_make_tables.py --output_dir tables/main
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Importing as `python scripts/main_make_tables.py` puts scripts on
# sys.path so the tables/ subpackage resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Active-artefact entry points (main-paper excerpt tables).
from tables.main.exp1_2 import build as build_exp1_2_main
from tables.main.exp3_1 import build as build_exp3_1_main
from tables.main.exp2_1_mak import build as build_exp2_1_main
from tables.main.exp2_2_kepler import build as build_exp2_2_main
from tables.main.exp2_3_species import build as build_exp2_3_species_main
from tables.main.exp3_3_archinv import build as build_exp3_3_main
from tables.main.exp3_3_archinv_mini import build as build_exp3_3_mini
from tables.main.histopath import build as build_histopath_main

# Re-export shared loaders + formatters for any external callers (and for the
# legacy ad-hoc builders kept in this file).
from tables._loaders import (
    load_exp2_3_species_clean,
    load_exp3_3_results,
    load_histopath,
    load_results,
)
from tables._fmt import agg, fmt, lbl
from tables.main._helpers import cond_metric, dgp_metric, fmt_or_dash

# Legacy builders below use the old name `_dgp_metric` for the same helper.
_dgp_metric = dgp_metric


# Dispatch maps — dispatched in the order listed by main().
MAIN_PAPER_BUILDERS = {
    "1.2": build_exp1_2_main,
    "2.1": build_exp2_1_main,
    "2.2": build_exp2_2_main,
    "3.1": build_exp3_1_main,
}


# ---------------------------------------------------------------------------
# Per-condition aggregators
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Cut builders
# ---------------------------------------------------------------------------

def table_exp1_1_main(runs: list[dict]) -> str:
    """Exp 1.1 excerpt: 5 rows. raw_mlp, tabpfn (raw), fourier_ols, fourier_mlp, tabpfn_fourier."""
    rows = []

    def addrow(fmap, model, cond, shaded=False, bold_ood=False):
        ide = cond_metric(runs, cond, "id_error")
        ood = cond_metric(runs, cond, "ood_error")
        if ood is None:
            return
        prefix = r"\rowcolor{gray!12}" + "\n    " if shaded else ""
        rows.append(
            f"    {prefix}{fmap} & {model} & "
            f"{fmt_or_dash(ide)} & {fmt_or_dash(ood, bold=bold_ood)} \\\\"
        )

    # Best-OOD detection so we can bold it
    cands = [("raw_mlp",), ("tabpfn",), ("fourier_ols",), ("fourier_mlp",), ("tabpfn_fourier",)]
    best = None
    for (k,) in cands:
        v = cond_metric(runs, k, "ood_error")
        if v is not None and (best is None or v[0] < best):
            best = v[0]

    addrow(r"None (raw $x$)", "MLP",     "raw_mlp",        shaded=False,
           bold_ood=(cond_metric(runs, "raw_mlp",        "ood_error") or (None,))[0] == best)
    addrow(r"None (raw $x$)", "TabPFN",  "tabpfn",         shaded=False,
           bold_ood=(cond_metric(runs, "tabpfn",         "ood_error") or (None,))[0] == best)
    addrow(r"Fourier $(\sin x, \cos x)$", "OLS",    "fourier_ols",     shaded=True,
           bold_ood=(cond_metric(runs, "fourier_ols",    "ood_error") or (None,))[0] == best)
    addrow(r"Fourier $(\sin x, \cos x)$", "MLP",    "fourier_mlp",     shaded=True,
           bold_ood=(cond_metric(runs, "fourier_mlp",    "ood_error") or (None,))[0] == best)
    addrow(r"Fourier $(\sin x, \cos x)$", "TabPFN", "tabpfn_fourier",  shaded=True,
           bold_ood=(cond_metric(runs, "tabpfn_fourier", "ood_error") or (None,))[0] == best)

    cap = (r"Exp.~1.1 (excerpt). Same training data on $W = [-\pi, \pi]$, OOD on $[\pi, 3\pi]$, "
           r"true DGP $f(x) = \sin(x)$. \colorbox{gray!12}{\strut Shaded rows} commit to the "
           r"periodic DGP via the Fourier feature map. OOD relative error (\%), "
           r"mean $\pm$ std over 3 seeds; best OOD in bold.")

    return (
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        f"  \\caption{{{cap}}}\n  \\label{{tab:exp11_main}}\n"
        "  \\begin{tabular}{lllr}\n    \\toprule\n"
        "    \\textbf{Feature map} & \\textbf{Model} & \\textbf{ID (\\%)} & \\textbf{OOD (\\%)} \\\\\n    \\midrule\n"
        + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    )






def table_exp1_3_sindy_main(runs: list[dict]) -> str:
    """Exp 1.3 SINDy diagnostic excerpt: 2 rows per process (correct + 1 wrong baseline).

    The full-battery builder consumes a different result shape (`records`, separate exp id);
    the main-paper version emits a hand-curated extract from the verified diagnostic battery
    (Spearman ρ=0.82, p=1.3e-9, n=35 — consistent across seeds). Process labels repeat on
    every row, the canonically-correct row is shaded.
    """
    cap = (
        r"SINDy $\delta_{\mathrm{OOD}}$ diagnostic (excerpt). For each (target "
        r"process, feature map) pair we fit a degree-2 polynomial SINDy model "
        r"in the transformed state and report the OOD derivative residual "
        r"$\delta_{\mathrm{OOD}}$. \colorbox{gray!12}{\strut Shaded rows} use "
        r"the canonically-correct coordinate; one wrong baseline per process "
        r"is shown for contrast. Spearman $\rho(\delta_{\mathrm{OOD}}, "
        r"\text{wrong-label}) = 0.82$, $p = 1.3{\times}10^{-9}$, $n=35$ pairs "
        r"in the full battery. $G_2 = \tanh(5\sin x \cos 2x)$ has no sparse "
        r"polynomial coordinate (diagnostic silent). Full grid in "
        r"App.~\ref{app:sindy_battery}."
    )

    # Hand-curated extract from the verified diagnostic battery.
    # Per block: (process label, [(feature_map, k, δ_tr, δ_ood, is_canonical)]).
    blocks = [
        (r"$\sin x$", [
            (r"Fourier-2 (deg.\ 1)",  4, 0.0000, 0.0000, True),
            (r"Poly-2",               2, 0.0000, 54.31,  False),
        ]),
        (r"$\sin(2x)$", [
            (r"Fourier-2 (deg.\ 2)",  4, 0.0000, 0.0000, True),
            (r"Poly-2",               3, 0.0003, 332.0,  False),
        ]),
        (r"$G_1 = \sin x \cos 2x$", [
            (r"Fourier-4 (deg.\ 2)",  3, 0.0000, 0.0000, True),
            (r"Poly-3",               2, 0.9638, 26.46,  False),
        ]),
        (r"$I_1\!:\, \log y = 2u-u^2$", [
            (r"Log-log",              2, 0.0000, 0.0000, True),
            (r"Poly-2 $x{\to}y$",     3, 0.0041, 9587.0, False),
        ]),
        (r"$I_2\!:\, \log y = 1-2e^{-x}$", [
            (r"Log-$y$, feat.\ $e^{-x}$", 1, 0.0000, 0.0000, True),
            (r"Poly-2 $x{\to}y$",         3, 0.0008, 1660.0, False),
        ]),
    ]

    body = []
    for i, (proc, rows) in enumerate(blocks):
        if i > 0:
            body.append("    \\midrule")
        for fmap, k, dtr, dood, bold in rows:
            if bold:
                dood_str = f"\\textbf{{{dood:.4f}}}"
            elif dood >= 100:
                dood_str = f"{dood:.0f}"
            else:
                dood_str = f"{dood:.2f}"
            shade = "    \\rowcolor{gray!12}\n" if bold else ""
            body.append(
                f"{shade}    {proc} & {fmap} & {k} & {dtr:.4f} & {dood_str} \\\\"
            )

    return (
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        f"  \\caption{{{cap}}}\n  \\label{{tab:exp13sindy_main}}\n"
        "  \\begin{tabular}{llrrr}\n    \\toprule\n"
        "    \\textbf{Process} & \\textbf{Feature map} & $\\boldsymbol{k}$ & "
        "$\\boldsymbol{\\delta_{\\mathrm{tr}}}$ & $\\boldsymbol{\\delta_{\\mathrm{OOD}}}$ \\\\\n"
        "    \\midrule\n"
        + "\n".join(body) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    )












# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir",  default="tables/main")
    parser.add_argument("--missing_ok",  action="store_true",
                        help="If results for a builder are missing, write a placeholder rather than skip")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating main-paper excerpt tables → {out}/")

    grouped = load_results(args.results_dir)

    # exp_id-driven builders (mirrors original MAIN_PAPER_BUILDERS layout).
    for exp_id, build_fn in MAIN_PAPER_BUILDERS.items():
        runs = grouped.get(exp_id, [])
        if not runs and not args.missing_ok:
            print(f"  [skip] {exp_id}: no results in {args.results_dir}")
            continue
        try:
            build_fn(runs, args.output_dir)
        except Exception as e:
            print(f"  [error] {exp_id}: {e}")
            continue

    # CDS species (specialized loader)
    cds_sp = load_exp2_3_species_clean(args.results_dir)
    if cds_sp:
        build_exp2_3_species_main(cds_sp, args.output_dir)
    else:
        print("  [skip] exp2_3 species: no results")

    # Camelyon17 histopathology (specialized loader)
    histo = load_histopath(args.results_dir)
    if histo:
        build_histopath_main(histo, args.output_dir)
    else:
        print("  [skip] histopath: no results")

    # Arch-Inv (specialized loader)
    archinv = load_exp3_3_results(args.results_dir)
    if archinv:
        build_exp3_3_main(archinv, args.output_dir)
        build_exp3_3_mini(archinv, args.output_dir)
    else:
        print("  [skip] exp3_3 archinv: no results")

    print("Done. Run sync_paper.py to copy into paper/tables/.")


if __name__ == "__main__":
    main()
