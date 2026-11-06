#!/usr/bin/env python3
"""Exp 4.5 (E3): phi-misspecification sweep on the torus benchmark.

Three independent ways of degrading the exact map phi(x) = (sin x_j, cos x_j),
one block each:

  A  wrong Fourier frequency   phi_omega(x) = (sin omega x_j, cos omega x_j),
                               omega_true = 1, d = 2. `raw` (no phi) anchor.
  B  partial coordinate cover  exact phi on the first m of d = 4 coordinates,
                               raw pass-through for the rest. m = 0 IS the raw
                               (no-commitment) anchor; m = 4 is exact.
  C  noisy phi                 exact frequencies, additive N(0, s^2) feature
                               noise, d = 2. s = 0 is exact.

Story: A and C degrade gracefully; B is NON-MONOTONE, and partial coverage
(m = 1, 2, 3) is worse than no commitment at all, because the target couples all
d coordinates multiplicatively.

Run as a script:
    conda run -n data_analysis python scripts/make_tables.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import math
import sys
from pathlib import Path

import numpy as np

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _loaders import load_results
from _fmt import (
    _latex_thousands,
    agg,
    fmt,
    table_footer,
    table_header,
)

EID = 'E3_misspec'
OUT = Path(__file__).resolve().parents[2] / "tables" / "table_e3_misspec.tex"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "results"

N_COLS = 3
SHADE = r"\rowcolor{gray!12}"
DEC = 4


def _load_input(results_dir=None):
    grouped = load_results(str(results_dir or RESULTS_DIR_DEFAULT))
    return grouped.get(EID, [])


def _predict_mean_rmse(lam, d):
    """Std of g_int(theta) = (1-lam)*mean_j sin(theta_j) + lam*prod_j sin(theta_j)
    for theta uniform on the d-torus: the predict-the-mean RMSE reference.
    Var = (1-lam)^2 * (1/2)/d + lam^2 * (1/2)^d (the two terms are uncorrelated).
    """
    return math.sqrt((1.0 - lam) ** 2 * 0.5 / d + lam ** 2 * 0.5 ** d)


def _cell(runs, axis, key):
    """(mean, std) of ID and OOD RMSE across seeds for one leaf, or None."""
    id_vals, ood_vals = [], []
    for r in runs:
        leaf = r.get("axes", {}).get(axis, {}).get(key)
        if not leaf:
            continue
        id_vals.append(float(leaf["id_rmse"]))
        ood_vals.append(float(leaf["ood_rmse"]))
    if not ood_vals:
        return None
    return agg(id_vals), agg(ood_vals)


# %% render — returns tex string (or None when no data)
def render(runs=None):
    if runs is None:
        runs = _load_input()
    if not runs:
        print("  [table_e3_misspec] no input found, skipping")
        return None

    cfg = runs[0].get("config", {})
    lam = cfg.get("lam", 0.35)
    d_a = cfg.get("d_axis_a", 2)
    d_b = cfg.get("d_axis_b", 4)
    omegas = cfg.get("omegas", [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.25, 1.50])
    noise_s = cfg.get("noise_s", [0.0, 0.05, 0.1, 0.2, 0.4])
    n_train = cfg.get("n_train", 4096)
    epochs = cfg.get("epochs", 3000)
    n_seeds = len(runs)

    ref_a = _predict_mean_rmse(lam, d_a)
    ref_b = _predict_mean_rmse(lam, d_b)

    # Per block: (header, [(row_label, axis, key, is_exact, is_raw_anchor)]).
    blocks = []

    rows_a = [(rf"$\omega = {om:.2f}$", "wrong_omega", f"omega_{om:.2f}",
               abs(om - 1.0) < 1e-9, False) for om in omegas]
    rows_a.append((r"\texttt{raw} (no $\varphi$)", "wrong_omega", "raw", False, True))
    blocks.append((
        rf"A. Wrong Fourier frequency ($d{{=}}{d_a}$, $\omega_\mathrm{{true}}{{=}}1$)",
        rows_a,
    ))

    rows_b = []
    for m in range(d_b + 1):
        if m == 0:
            label = rf"$m = 0$ (\texttt{{raw}}, no $\varphi$)"
        elif m == d_b:
            label = rf"$m = {m}$ (exact $\varphi$)"
        else:
            label = rf"$m = {m}$"
        rows_b.append((label, "partial_phi", f"m_{m}", m == d_b, m == 0))
    blocks.append((
        rf"B. Partial coordinate coverage ($\varphi$ on $m$ of $d{{=}}{d_b}$)",
        rows_b,
    ))

    rows_c = [(rf"$s = {s:.2f}$" + (r" (exact $\varphi$)" if s == 0 else ""),
               "noisy_phi", f"s_{s:.2f}", s == 0, False) for s in noise_s]
    blocks.append((
        rf"C. Noisy $\varphi$ (feature noise $\mathcal{{N}}(0, s^2)$, $d{{=}}{d_a}$)",
        rows_c,
    ))

    cap = (
        r"Exp.~4.5: $\varphi$-misspecification sweep on the torus benchmark. "
        r"Latent $\theta \in \mathbb{T}^d$, training on the $k{=}0$ block, OOD "
        r"by an integer block shift $x = \theta + 2\pi$; target "
        r"$g_\mathrm{int}$ with $\lambda = " + f"{lam}" + r"$, MLP "
        r"($5$ Tanh layers, $256$ units), "
        rf"$N_\mathrm{{train}} = {_latex_thousands(n_train)}$, "
        rf"${_latex_thousands(epochs)}$ epochs. "
        r"Three axes degrade the exact map "
        r"$\varphi(x) = (\sin x_j, \cos x_j)_{j=1}^d$ independently. "
        r"\textbf{A}: a wrong Fourier frequency, "
        r"$\varphi_\omega(x) = (\sin \omega x_j, \cos \omega x_j)_{j=1}^d$. "
        r"\textbf{B}: the exact map on only the first $m$ coordinates, raw "
        r"pass-through for the rest, so $m = 0$ is itself the no-commitment "
        r"anchor. \textbf{C}: exact frequencies but additive feature noise. "
        r"Rows labelled \texttt{raw} carry no commitment at all and are the "
        r"reference anchor; for scale, the predict-the-mean RMSE of the target "
        rf"is ${ref_a:.3f}$ at $d{{=}}{d_a}$ and ${ref_b:.3f}$ at "
        rf"$d{{=}}{d_b}$. "
        rf"ID and OOD RMSE, mean $\pm$ std over {n_seeds} seeds; best OOD RMSE "
        r"per block in bold; \colorbox{gray!12}{\strut shaded rows} use the "
        r"exact map. "
        r"Degradation is graceful on the frequency and noise axes (A and C: "
        r"OOD RMSE interpolates smoothly between the exact commitment and the "
        r"no-commitment anchor, with no cliff), but non-monotone on the "
        r"coordinate-coverage axis (B): because $g_\mathrm{int}$ couples all "
        r"$d$ coordinates multiplicatively, partial coverage "
        r"($m = 1, 2, 3$) is \emph{worse} than no commitment at all. "
        r"Partial alignment buys partial extrapolation; partial \emph{coverage} "
        r"of a coupled mechanism buys none."
    )

    out = table_header(
        cap, "tab:e3_misspec", "lrr",
        r"\textbf{Setting} & \textbf{ID RMSE} & \textbf{OOD RMSE}"
    )

    for bi, (block_head, rows) in enumerate(blocks):
        if bi > 0:
            out += "    \\midrule\n"
        out += f"    \\multicolumn{{{N_COLS}}}{{l}}{{\\emph{{{block_head}}}}} \\\\\n"
        out += "    \\midrule\n"

        cells = {key: _cell(runs, axis, key) for _, axis, key, _, _ in rows}
        finite = [c[1][0] for c in cells.values() if c is not None]
        best = min(finite) if finite else None

        for label, axis, key, is_exact, is_raw in rows:
            c = cells[key]
            if c is None:
                out += f"    {label} & --- & --- \\\\\n"
                continue
            (id_m, id_s), (ood_m, ood_s) = c
            is_best = best is not None and abs(ood_m - best) < 1e-12
            prefix = f"    {SHADE}\n    " if is_exact else "    "
            out += (f"{prefix}{label} & {fmt(id_m, id_s, decimals=DEC)} & "
                    f"{fmt(ood_m, ood_s, bold=is_best, decimals=DEC)} \\\\\n")

    out += table_footer()
    return out


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
