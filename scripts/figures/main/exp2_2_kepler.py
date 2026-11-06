"""Main-paper Kepler concept: iso-T contours, log-OLS vs raw-MLP.

Single panel in raw $(a, M_\\star)$ space. ID region (a < 0.5 AU) shaded blue,
OOD region (0.5 <= a < 20) shaded red, intensity = true T(a, M_star).
Dense iso-T contour overlay: log-OLS prediction (green solid) overlays the
true levels everywhere; raw-MLP prediction (pink dashed) bends the wrong way
in the OOD region. Same MAK punch: linear-in-(log a, log M_star) recovers a
highly non-linear surface in raw (a, M_star, T) space; raw MLP fails OOD.

Run as a script:
    conda run -n data_analysis python scripts/main_plot_concept.py
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import RC

OUT = Path(__file__).resolve().parents[3] / "figures" / "main" / "exp2_2_concept_main.pdf"
CACHE = Path(__file__).resolve().parents[3] / "data" / "kepler_clean.parquet"

A_SPLIT = 0.5
A_VIS = 20.0

BLUE = "#2196F3"
PINK = "#E91E8C"
OLS_GREEN = "#2ecc71"


# %% data + models
@lru_cache(maxsize=1)
def _load_kepler():
    """Load the NASA exoplanets dataset and return raw (a, T, M_star) arrays.

    Memoised at module scope: re-running the render cell in the same kernel
    reuses the loaded arrays. Cleaned subset is also persisted to a local
    parquet so a fresh kernel skips the HuggingFace roundtrip.
    """
    import pandas as pd
    if CACHE.exists():
        sub = pd.read_parquet(CACHE)
    else:
        from datasets import load_dataset
        ds = load_dataset("juliensimon/nasa-exoplanets", split="train")
        df = ds.to_pandas()
        sub = df[["pl_orbsmax", "pl_orbper", "st_mass"]].dropna()
        sub = sub[(sub["pl_orbsmax"] > 0) & (sub["pl_orbper"] > 0) & (sub["st_mass"] > 0)]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(CACHE, index=False)
    return (sub["pl_orbsmax"].values,
            sub["pl_orbper"].values / 365.25,
            sub["st_mass"].values)


def _split(a, T, M):
    m_tr = a < A_SPLIT
    m_od = (a >= A_SPLIT) & (a < A_VIS)
    return dict(
        a_tr=a[m_tr], T_tr=T[m_tr], M_tr=M[m_tr],
        a_od=a[m_od], T_od=T[m_od], M_od=M[m_od],
    )


def _fit_models(d):
    la = np.log(d["a_tr"]); lM = np.log(d["M_tr"]); lT = np.log(d["T_tr"])
    A_log = np.column_stack([la, lM, np.ones_like(la)])
    w, *_ = np.linalg.lstsq(A_log, lT, rcond=None)
    b_a, b_M, c = float(w[0]), float(w[1]), float(w[2])

    def predict_log_ols(A_grid, M_grid):
        return np.exp(b_a * np.log(A_grid) + b_M * np.log(M_grid) + c)

    X_tr = np.column_stack([d["a_tr"], d["M_tr"]])
    sx = StandardScaler().fit(X_tr)
    sy_mean = float(d["T_tr"].mean())
    sy_std = float(d["T_tr"].std())
    y = (d["T_tr"] - sy_mean) / sy_std
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="tanh",
        max_iter=8000,
        random_state=42,
        early_stopping=False,
    ).fit(sx.transform(X_tr), y)

    def predict_raw_mlp(A_grid, M_grid):
        shape = A_grid.shape
        X = np.column_stack([A_grid.ravel(), M_grid.ravel()])
        Xn = sx.transform(X)
        yhat = mlp.predict(Xn) * sy_std + sy_mean
        return yhat.reshape(shape)

    return predict_log_ols, predict_raw_mlp, (b_a, b_M, c)


def _make_grid(a_lo=0.01, a_hi=A_VIS, M_lo=0.3, M_hi=3.0, n_a=160, n_M=160):
    ga = np.logspace(np.log10(a_lo), np.log10(a_hi), n_a)
    gM = np.linspace(M_lo, M_hi, n_M)
    A, M = np.meshgrid(ga, gM, indexing="xy")
    T_true = A**1.5 / np.sqrt(M)
    return ga, gM, A, M, T_true


# %% render — pure: builds and returns fig, no save, no close
def render(data=None):
    if data is None:
        a_all, T_all, M_all = _load_kepler()
    else:
        a_all, T_all, M_all = data
    d = _split(a_all, T_all, M_all)

    predict_log_ols, predict_raw_mlp, _ = _fit_models(d)
    ga, gM, A, M, T_true = _make_grid()
    T_ols = predict_log_ols(A, M)
    T_mlp = predict_raw_mlp(A, M)

    LEVELS = np.geomspace(1e-3, 3e2, 40)
    cmap_id = mcolors.LinearSegmentedColormap.from_list(
        "id_blues", ["#cfe5f5", "#1976d2"])
    cmap_ood = mcolors.LinearSegmentedColormap.from_list(
        "ood_reds", ["#fad5d5", "#d32f2f"])
    bg_norm = mcolors.LogNorm(vmin=1e-3, vmax=3e2)

    rc = {
        **RC,
        "axes.grid":       False,
        "axes.labelsize":  20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0),
                               constrained_layout=True)
        ax.set_box_aspect(1)

        ax.set_xscale("log")
        ax.set_xlim(ga[0], ga[-1])
        ax.set_ylim(gM[0], gM[-1])
        ax.set_xlabel(r"$a$  [AU]")
        ax.set_ylabel(r"$M_\star$  [$M_\odot$]")

        # Split shaded background. Boundary column is in both arrays so OOD
        # (drawn second) covers it cleanly with no transparent seam.
        boundary_i = int(np.searchsorted(ga, A_SPLIT))
        T_id_arr = T_true.copy(); T_id_arr[:, boundary_i + 1:] = np.nan
        T_ood_arr = T_true.copy(); T_ood_arr[:, :boundary_i] = np.nan
        ax.pcolormesh(ga, gM, T_id_arr, shading="auto",
                      cmap=cmap_id, norm=bg_norm, alpha=0.92, zorder=1,
                      rasterized=True)
        ax.pcolormesh(ga, gM, T_ood_arr, shading="auto",
                      cmap=cmap_ood, norm=bg_norm, alpha=0.92, zorder=2,
                      rasterized=True)

        # Faint true-T iso-contours for reference (visual scaffold).
        ax.contour(A, M, T_true, levels=LEVELS,
                   colors="white", linewidths=0.3, alpha=0.5, zorder=3)
        # log-OLS contours overlay truth exactly.
        ax.contour(A, M, T_ols, levels=LEVELS,
                   colors=OLS_GREEN, linewidths=1.4, alpha=0.95, zorder=4)
        # raw-MLP contours deviate OOD.
        ax.contour(A, M, T_mlp, levels=LEVELS,
                   colors=PINK, linewidths=1.4, linestyles="--",
                   alpha=0.95, zorder=5)

        # Train / OOD scatter (thinned).
        rng = np.random.default_rng(42)
        idx_tr = rng.choice(len(d["a_tr"]),
                            size=min(400, len(d["a_tr"])), replace=False)
        idx_od = rng.choice(len(d["a_od"]),
                            size=min(200, len(d["a_od"])), replace=False)
        ax.scatter(d["a_tr"][idx_tr], d["M_tr"][idx_tr],
                   c="black", s=50, marker="+", linewidths=1.0,
                   alpha=0.7, zorder=6)
        ax.scatter(d["a_od"][idx_od], d["M_od"][idx_od],
                   facecolors="none", edgecolors="black",
                   s=50, marker="^", linewidths=1.0, alpha=0.85, zorder=6)

        bbox = dict(facecolor="white", alpha=0.85, edgecolor="none",
                    boxstyle="round,pad=0.25")
        ax.text(0.02, 0.97, r"ID (train)  $a < 0.5$ AU",
                transform=ax.transAxes, fontsize=15, va="top", ha="left",
                color="#1f6091", bbox=bbox, zorder=10)
        ax.text(0.75, 0.97, r"OOD  $0.5 \leq a < 20$ AU",
                transform=ax.transAxes, fontsize=15, va="top", ha="center",
                color="#9b1f1f", bbox=bbox, zorder=10)

        legend_elements = [
            Line2D([0], [0], color=OLS_GREEN, lw=1.5,
                   label=r"log-OLS  (overlays true)"),
            Line2D([0], [0], color=PINK, lw=1.5, ls="--",
                   label=r"raw-MLP  (deviates OOD)"),
            Line2D([0], [0], color="black", marker="+", lw=0,
                   markersize=5, markeredgewidth=0.6, label="train pts"),
            Line2D([0], [0], color="black", marker="^", lw=0,
                   markersize=4, markeredgewidth=0.6,
                   markerfacecolor="none", label="OOD pts"),
        ]
        leg = ax.legend(handles=legend_elements, fontsize=15,
                  loc="upper left", borderpad=0.3,
                  markerscale=2,
                  handlelength=0.8, handletextpad=0.4, labelspacing=0.4,
                  frameon=True, bbox_to_anchor=(-0.01, 0.92))
    return fig


# %% build — CLI / shell entry point
def build(output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = render()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    fig = render()
    fig
# %%
