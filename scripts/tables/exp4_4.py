#!/usr/bin/env python3
"""Exp 4.4 (Tor-Noise): OOD RMSE and OOD R^2 per (d, σ) for raw / learned-Fourier
/ exact-φ.

One block per d. Rows are σ; the left column group is OOD RMSE (best per row
bolded) and the right column group is OOD R^2 (bolded at ≥ 0.95, matching the
sibling torus tables `exp4_2` / `exp4_3`). The two metrics answer different
questions and are reported side by side for that reason: R^2 is an absolute
"does this arm beat predicting the mean of the clean OOD target" test, whose
denominator is the *clean* OOD target variance and therefore identical across
arms and across σ, so it goes negative for every arm once the noise floor
exceeds the target's own spread. The relative question ("how much does the
commitment buy over no commitment") is answered by the companion ratio table
`exp4_4_ratio`, which stays informative after every R^2 has gone negative.

The result instantiates Prop:erm and Lem:canon: the canonical commitment is
σ-robust (its OOD error is Bayes-only), the wrong commitment is bias-dominated
and σ-independent.

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
"""

# %% imports
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _loaders import load_results
from _fmt import fmt

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
from exp4_4_regime_boundary import boundary_interval

EID = "4.4"
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_exp4_4.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"

CONDITIONS = ("raw_mlp", "learned_fourier_mlp", "exact_phi_mlp")
COND_SHORT = {
    "raw_mlp":             r"raw",
    "learned_fourier_mlp": r"learned-Fourier",
    "exact_phi_mlp":       r"exact-$\varphi$",
}


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


def _r2_decimals(mean, sig=3):
    """Decimal places giving `sig` significant digits on `mean`.

    OOD R^2 here spans 0.998 down to about -5e4, so a fixed `decimals=3` (the
    convention in exp4_2 / exp4_3, whose R^2 all sit in [-10, 1]) would print
    six meaningless digits in the high-σ rows. Fixing the significant digits
    instead keeps the column narrow and the precision honest.
    """
    if mean == 0 or not np.isfinite(mean):
        return 3
    return max(0, sig - 1 - int(math.floor(math.log10(abs(mean)))))


# %% render — returns tex string (or None when no data)
def noise_slopes(runs, cond="exact_phi_mlp"):
    """Mean log-log slope of OOD error against sigma, per d, over sigma > 0.

    Each run stores one slope per (d, arm) in its `bayes_slopes` field. This
    averages them over seeds, which is the quantity quoted for the canonical
    arm in the Exp. 4.4 discussion.
    """
    per_d = defaultdict(list)
    for run in runs:
        for d_key, arms in run.get("bayes_slopes", {}).items():
            if cond in arms:
                per_d[int(d_key.split("=")[1])].append(float(arms[cond]))
    return {d: float(np.mean(v)) for d, v in sorted(per_d.items()) if v}


def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print(f"  [table_exp4_4] no input found, skipping")
        return None

    # by_d_sigma[d][sigma][cond][metric] -> list over seeds
    by_d_sigma = defaultdict(lambda: defaultdict(
        lambda: {c: {"ood_rmse": [], "ood_r2": []} for c in CONDITIONS}))

    for run in runs:
        for cell in run["conditions"].values():
            d = int(cell["d"])
            sigma = float(cell["sigma"])
            for cond in CONDITIONS:
                by_d_sigma[d][sigma][cond]["ood_rmse"].append(
                    cell[cond]["ood_rmse"])
                by_d_sigma[d][sigma][cond]["ood_r2"].append(
                    cell[cond]["ood_r2"])

    ds = sorted(by_d_sigma.keys())
    n_seeds = len(runs)
    _, _, sigma_lo, sigma_hi, _ = boundary_interval(runs)

    slopes = noise_slopes(runs)
    slope_str = (rf" (log-log slopes {min(slopes.values()):.2f} to "
                 rf"{max(slopes.values()):.2f} over $\sigma>0$)" if slopes else "")

    caption = (
        r"Exp.~4.4: Noise robustness (d) of the DGP commitment on the "
        r"$d$-torus, fixed $N{=}4{,}096$. Training labels carry Gaussian noise "
        r"$\eta \sim \mathcal{N}(0, \sigma^2)$; ID and OOD targets remain "
        r"clean. The canonical $\varphi(x){=}(\sin x_j, \cos x_j)_{j=1}^d$ "
        r"shows approximately linear error scaling with training-noise level"
        + slope_str + r". Raw and learned-Fourier baselines are bias-floored "
        r"below the plug-in crossover from \Cref{prop:erm} "
        r"($\sigma \lesssim \sqrt{n}\,\eps_W/\sqrt{2\pi}$, "
        rf"bracketed by the stored cells to $[{sigma_lo:.2f}, {sigma_hi:.2f}]$) "
        r"and rise with $\sigma$ above this estimated crossover, but "
        r"the canonical commitment retains the OOD advantage throughout. "
        r"OOD RMSE, mean$\pm$std over " + f"{n_seeds}" +
        r" seeds; best OOD RMSE per row in \textbf{bold}. "
        # --- added for the R^2 column ---
        r"All three arms are the same MLP under three feature maps. "
        r"The two metrics answer different questions. OOD $R^2$ answers "
        r"\emph{does this arm beat predicting the mean, in absolute terms}: "
        r"it is positive only while the OOD RMSE stays below the standard "
        r"deviation of the clean OOD target, and goes negative once the RMSE "
        r"exceeds it. Its denominator is the \emph{clean} OOD target "
        r"variance, which is the same for every arm and every $\sigma$, so at "
        r"high $\sigma$ every arm is negative: no arm extrapolates usefully "
        r"in absolute terms there, however the arms rank against each other. "
        r"\Cref{tab:exp44_ratio} answers the complementary relative question. "
        r"OOD $R^2 \geq 0.95$ in \textbf{bold}."
    )

    head_top = (r"& \multicolumn{3}{c}{OOD RMSE} & "
                r"\multicolumn{3}{c}{OOD $R^2$}")
    cmid = r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}"
    short = [COND_SHORT[c] for c in CONDITIONS]
    head_sub = r"$\sigma$ & " + " & ".join(short + short)

    out = []
    out.append(
        f"\\begin{{table}}[ht]\n"
        f"  \\centering\n"
        f"  \\small\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{tab:exp44}}\n"
        f"  \\resizebox{{\\textwidth}}{{!}}{{%\n"
        f"  \\begin{{tabular}}{{lrrrrrr}}\n"
        f"    \\toprule\n"
        f"    {head_top} \\\\\n"
        f"    {cmid}\n"
        f"    {head_sub} \\\\\n"
        f"    \\midrule\n"
    )

    for i, d in enumerate(ds):
        if i > 0:
            out.append(r"    \midrule")
        out.append(rf"    \multicolumn{{7}}{{l}}{{\emph{{$d = {d}$}}}} \\")
        out.append(r"    \midrule")
        sigmas = sorted(by_d_sigma[d].keys())
        for sigma in sigmas:
            means = {}
            stds  = {}
            r2_means = {}
            r2_stds  = {}
            for cond in CONDITIONS:
                v = by_d_sigma[d][sigma][cond]["ood_rmse"]
                if v:
                    means[cond] = float(np.mean(v))
                    stds[cond]  = float(np.std(v))
                else:
                    means[cond] = float("nan")
                    stds[cond]  = float("nan")
                r = by_d_sigma[d][sigma][cond]["ood_r2"]
                if r:
                    r2_means[cond] = float(np.mean(r))
                    r2_stds[cond]  = float(np.std(r))
                else:
                    r2_means[cond] = float("nan")
                    r2_stds[cond]  = float("nan")
            best = min((m for m in means.values() if not np.isnan(m)),
                       default=float("inf"))
            cells = []
            for cond in CONDITIONS:
                is_best = (means[cond] == best)
                cells.append(fmt(means[cond], stds[cond],
                                 bold=is_best, decimals=3))
            for cond in CONDITIONS:
                cells.append(fmt(r2_means[cond], r2_stds[cond],
                                 bold=(r2_means[cond] >= 0.95),
                                 decimals=_r2_decimals(r2_means[cond])))
            sigma_str = f"{sigma:g}" if sigma > 0 else "0"
            out.append(f"    {sigma_str} & " + " & ".join(cells) + r" \\")

    out.append("    \\bottomrule\n  \\end{tabular}\n  }\n\\end{table}\n")
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
    for cond in CONDITIONS:
        s = noise_slopes(runs if runs is not None else _load_input(), cond)
        if s:
            print("    log-log slope of OOD error vs sigma (sigma>0), %-18s %s"
                  % (cond, "  ".join(f"d={d}: {v:.2f}" for d, v in s.items())))
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    tex = render()
    if tex:
        print(tex)
