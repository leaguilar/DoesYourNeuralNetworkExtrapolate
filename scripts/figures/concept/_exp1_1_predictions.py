"""Cache generator for exp1_1_concept overlay predictions.

Runs the four "from raw" baselines from exp1.1 (SINDy raw, MLP raw, TabPFN raw,
TimesFM raw) on the same training setup as `src/experiments/exp1_1_sin.py`
(seed=42, N_TRAIN=128, [-π, π] training, [π, 3π] OOD) and dumps predictions
into a single npz cache at `figures/_cache/exp1_1_predictions.npz` so the
concept-figure builder can overlay them without retraining each render.

Usage:
    # SINDy + MLP + TabPFN (data_analysis env)
    conda run -n data_analysis python \
        scripts/figures/concept/_exp1_1_predictions.py

    # TimesFM only (timesfm_env — Python compat)
    conda run -n timesfm_env python \
        scripts/figures/concept/_exp1_1_predictions.py --timesfm

    # Skip TabPFN (e.g. offline)
    conda run -n data_analysis python \
        scripts/figures/concept/_exp1_1_predictions.py --skip-tabpfn

The script merges new predictions into the existing cache, so the two env runs
can be done in any order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make src/experiments importable for utils / models
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parents[2]
sys.path.insert(0, str(_SRC / "experiments"))

CACHE_PATH = _SRC / "figures" / "_cache" / "exp1_1_predictions.npz"

N_TRAIN = 128
N_OOD   = 128
SEED    = 42
EPOCHS  = 3000
DENSE_N = 600   # plotting grid resolution across [-π, 3π]


def _build_grids():
    step = 2 * np.pi / (N_TRAIN - 1)
    x_all   = -np.pi + step * np.arange(N_TRAIN + N_OOD)
    x_train = x_all[:N_TRAIN].reshape(-1, 1)
    x_ood   = x_all[N_TRAIN:].reshape(-1, 1)
    x_dense = np.linspace(-np.pi, 3 * np.pi, DENSE_N).reshape(-1, 1)
    y_train = np.sin(x_train).flatten()
    return x_train, x_ood, x_dense, y_train


def _load_existing() -> dict:
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return {}


def _save(merged: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **merged)
    print(f"  Saved → {CACHE_PATH}")


def run_sindy_mlp_tabpfn(skip_tabpfn: bool = False) -> dict:
    from utils import set_seed, MLP, train_mlp, evaluate_mlp
    import pysindy as ps

    set_seed(SEED)
    x_train, x_ood, x_dense, y_train = _build_grids()

    out = dict(
        x_train=x_train.flatten(),
        y_train=y_train,
        x_ood=x_ood.flatten(),
        x_dense=x_dense.flatten(),
    )

    # ── SINDy raw (PolynomialLibrary deg=9, STLSQ) ─────────────────────────
    lib = ps.PolynomialLibrary(degree=9, include_bias=True)
    lib.fit(x_train)
    Th_tr  = np.asarray(lib.transform(x_train))
    Th_dn  = np.asarray(lib.transform(x_dense))
    opt = ps.STLSQ(threshold=1e-6, alpha=0.0, max_iter=50)
    opt.fit(Th_tr, y_train.reshape(-1, 1))
    xi = np.asarray(opt.coef_[0])
    out["sindy_raw_dense"] = (Th_dn @ xi).flatten()
    print(f"  SINDy raw: n_terms={int((np.abs(xi) > 1e-10).sum())}")

    # ── MLP raw (matches exp1_1_sin.py) ────────────────────────────────────
    set_seed(SEED)
    m = MLP(1)
    train_mlp(m, x_train, y_train, epochs=EPOCHS)
    out["mlp_raw_dense"] = evaluate_mlp(m, x_dense)
    print("  MLP raw: trained")

    # ── TabPFN raw ─────────────────────────────────────────────────────────
    if not skip_tabpfn:
        try:
            from models import TabPFNModel
            tpfn = TabPFNModel()
            if tpfn.available:
                print("  TabPFN raw: cloud inference …")
                tpfn.fit(x_train, y_train)
                out["tabpfn_raw_dense"] = tpfn.predict(x_dense)
            else:
                print("  TabPFN raw: SKIP (token not available)")
        except Exception as e:
            print(f"  TabPFN raw: SKIP ({e})")

    return out


def _taylor9(x):
    """sin(x) Taylor series around 0, degree 9 (matches _helpers.taylor9)."""
    return x - x**3 / 6 + x**5 / 120 - x**7 / 5040 + x**9 / 362880


def run_timesfm(backend: str = "cpu") -> dict:
    from models import TimesFMModel

    x_train, x_ood, x_dense, y_train = _build_grids()
    out = dict(
        x_train=x_train.flatten(),
        y_train=y_train,
        x_ood=x_ood.flatten(),
        x_dense=x_dense.flatten(),
    )

    print("  Loading TimesFM 1.0-200M …")
    tfm = TimesFMModel(context_len=N_TRAIN, horizon_len=N_OOD, backend=backend)

    # ── TimesFM raw on sin context ─────────────────────────────────────────
    tfm.set_context(y_train)
    out["timesfm_raw_ood"] = tfm.forecast(N_OOD)
    print("  TimesFM raw (sin context): forecast generated")

    # ── TimesFM raw on Taylor-9 context ────────────────────────────────────
    # Feed TimesFM the Taylor-9 evaluations on x_train as the sequence context,
    # then forecast OOD. Lets the figure show what a sequence forecaster does
    # when handed the polynomial-commitment signal instead of the true sin.
    y_t9_train = _taylor9(x_train.flatten())
    tfm.set_context(y_t9_train)
    out["timesfm_taylor9_ood"] = tfm.forecast(N_OOD)
    print("  TimesFM raw (Taylor-9 context): forecast generated")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesfm",     action="store_true",
                    help="Run only TimesFM (use timesfm_env).")
    ap.add_argument("--skip-tabpfn", action="store_true")
    ap.add_argument("--backend",     default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    merged = _load_existing()
    if args.timesfm:
        new = run_timesfm(backend=args.backend)
    else:
        new = run_sindy_mlp_tabpfn(skip_tabpfn=args.skip_tabpfn)
    merged.update(new)
    _save(merged)


if __name__ == "__main__":
    main()
