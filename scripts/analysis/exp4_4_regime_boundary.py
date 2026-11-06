#!/usr/bin/env python3
"""Where the two regimes of Prop. 1 meet on the d-torus (Exp. 4.4).

Prop. 1 bounds the worst-case correct-decision probability of any W-supported
test by Phi(t_W / 2) with t_W = sqrt(n) eps_W / sigma, and its linear relaxation

    1/2 + eps_W sqrt(n) / (2 sqrt(2 pi) sigma)

reaches 1 -- i.e. stops constraining any test at all -- exactly at

    sigma_star = sqrt(n) eps_W / sqrt(2 pi).                              (*)

Locating that boundary on the torus sweep needs eps_W, the in-window L2 distance
between the process the raw commitment settles on and the one the canonical
(exact-phi) commitment settles on. Exp. 4.4 stores three L2/L1 scalars per cell
on a common clean ID grid:

    eps_observed             = mean |raw - exact|          (L1)
    raw_mlp.id_rmse          = || raw   - y_id ||_L2
    exact_phi_mlp.id_rmse    = || exact - y_id ||_L2

which bracket the L2 quantity exactly, with no further assumptions:

    lower:  max( mean|raw - exact|,  | rmse_raw - rmse_exact | )  <=  eps_W
            (Jensen; reverse triangle inequality)
    upper:  eps_W  <=  rmse_raw + rmse_exact
            (triangle inequality)

Both bounds are computed from the stored sweep, so the regime boundary is an
interval rather than a point estimate. Cells are restricted to 0 < sigma <= 0.05
(the near-noiseless end of the sweep), where the gap measures the commitment and
not the noise the two models are fitting.

Writes results/analysis/exp4_4_regime_boundary.json.

Run as a script (from the project root):
    conda run -n data_analysis python scripts/analysis/exp4_4_regime_boundary.py
"""

# %% imports
import json
import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
_FIGRES = _SRC / "scripts" / "figures" / "results"
if str(_FIGRES) not in sys.path:
    sys.path.insert(0, str(_FIGRES))

from _data import load_results  # noqa: E402

EID = "4.4"
RESULTS_DIR_DEFAULT = _SRC / "results"
OUT = _SRC / "results" / "analysis" / "exp4_4_regime_boundary.json"

# Cells used for the commitment gap: noise present but negligible.
SIGMA_MAX = 0.05


# %% bracket
def eps_w_bracket(runs):
    """Per-cell lower/upper bounds on the in-window L2 commitment gap eps_W."""
    lo, hi, n_train = [], [], None
    for run in runs:
        n_train = run.get("n_train", n_train)
        for cell in run["conditions"].values():
            sigma = float(cell["sigma"])
            if not (0 < sigma <= SIGMA_MAX):
                continue
            l1 = float(cell["eps_observed"])
            r_raw = float(cell["raw_mlp"]["id_rmse"])
            r_ex = float(cell["exact_phi_mlp"]["id_rmse"])
            lo.append(max(l1, abs(r_raw - r_ex)))
            hi.append(r_raw + r_ex)
    return np.asarray(lo), np.asarray(hi), int(n_train or 4096)


def sigma_star(eps_w, n):
    """Boundary (*) at which the linear form of Prop. 1 stops constraining."""
    return math.sqrt(n) * eps_w / math.sqrt(2.0 * math.pi)


def boundary_interval(runs):
    """(eps_lo, eps_hi, sigma_lo, sigma_hi, n) from the median cell bounds."""
    lo, hi, n = eps_w_bracket(runs)
    e_lo, e_hi = float(np.median(lo)), float(np.median(hi))
    return e_lo, e_hi, sigma_star(e_lo, n), sigma_star(e_hi, n), n


# %% report
def main(results_dir=None):
    runs = load_results(str(results_dir or RESULTS_DIR_DEFAULT)).get(EID, [])
    if not runs:
        print("  [4.4] no runs found")
        return None
    lo, hi, n = eps_w_bracket(runs)
    e_lo, e_hi, s_lo, s_hi, n = boundary_interval(runs)
    data = dict(
        experiment_id = EID,
        n_seeds       = len(runs),
        n_cells       = int(lo.size),
        n_train       = n,
        sigma_max_used= SIGMA_MAX,
        eps_w_lower   = e_lo,
        eps_w_upper   = e_hi,
        sigma_star_lower = s_lo,
        sigma_star_upper = s_hi,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Exp 4.4: {len(runs)} seeds, {lo.size} cells with 0 < sigma <= {SIGMA_MAX}, "
          f"n_train = {n}")
    print(f"    eps_W  in [{e_lo:.4f}, {e_hi:.4f}]   (median per-cell bounds)")
    print(f"    sigma* in [{s_lo:.3f}, {s_hi:.3f}]   = sqrt(n) eps_W / sqrt(2 pi)")
    print(f"  Saved -> {OUT}")
    return data


if __name__ == "__main__":
    main()
