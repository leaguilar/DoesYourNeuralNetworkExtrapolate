"""Shared data-loading and aggregation helpers for results figures."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

# Make _style importable as a top-level module when this file is loaded as a
# script (e.g. in VS Code 'Run Cell' mode where __package__ is not set).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import col, lbl


def load_results(results_dir: str) -> dict[str, list[dict]]:
    """Load JSON files; keep only the latest file per (exp_id, seed).

    Searches both the flat layout (legacy) and per-experiment subdirs.
    """
    root = Path(results_dir)
    files = sorted(root.glob("exp*.json")) + sorted(root.glob("*/exp*.json"))
    by_key = {}
    for p in files:
        parts = p.stem.split("__")
        if len(parts) < 3:
            continue
        exp_part  = parts[0]
        seed_part = parts[1]
        by_key[(exp_part, seed_part)] = p

    grouped = defaultdict(list)
    for p in by_key.values():
        with open(p) as f:
            data = json.load(f)
        grouped[data["experiment_id"]].append(data)
    return dict(grouped)


def agg_conditions(runs: list[dict]) -> dict[str, dict]:
    """Mean ± std over seeds for experiments with a top-level 'conditions' dict."""
    all_conds = list(runs[0]["conditions"].keys())
    result = {}
    for cond in all_conds:
        metrics = defaultdict(list)
        for run in runs:
            cdata = run["conditions"][cond]
            for k, v in cdata.items():
                if isinstance(v, (int, float)):
                    metrics[k].append(v)
        result[cond] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for k, v in metrics.items()}
    return result


def hbar(ax, conditions, id_means, ood_means, id_stds=None, ood_stds=None,
         colors=None, labels=None, title="", extra_handles=None, show_legend=True):
    """Horizontal grouped bar: ID (hatched) and OOD (solid) per condition."""
    n = len(conditions)
    y = np.arange(n)
    h = 0.35
    clrs  = colors or [col(c) for c in conditions]
    lbls  = labels or [lbl(c) for c in conditions]

    for i, (c_val, color) in enumerate(zip(conditions, clrs)):
        id_m  = id_means[i]  if id_means  is not None else None
        ood_m = ood_means[i]
        id_s  = (id_stds[i]  if id_stds  is not None else 0) if id_means is not None else 0
        ood_s = ood_stds[i]  if ood_stds is not None else 0

        if id_means is not None:
            ax.barh(y[i] + h / 2, id_m, xerr=id_s, color=color, height=h,
                    alpha=0.4, hatch="//", edgecolor=color, capsize=2,
                    error_kw={"elinewidth": 0.8, "ecolor": "black"})
        ax.barh(y[i] - h / 2 if id_means is not None else y[i],
                ood_m, xerr=ood_s, color=color, edgecolor="none",
                height=h, alpha=0.85, capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "black"})

    ax.set_yticks(y)
    ax.set_yticklabels(lbls)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Error (%)")
    ax.set_title(title, pad=4)

    if show_legend and id_means is not None:
        handles = [mpatches.Patch(facecolor=c, alpha=0.85, edgecolor="none", label=l)
                   for c, l in zip(clrs, lbls)]
        handles += [
            mpatches.Patch(facecolor="dimgrey", alpha=0.4, hatch="//",
                           edgecolor="dimgrey", label="In-dist.  (hatched)"),
            mpatches.Patch(facecolor="dimgrey", alpha=0.85, edgecolor="none",
                           label="OOD  (solid)"),
        ]
        if extra_handles:
            handles += extra_handles
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")


def tor_aggseeds(values_per_seed):
    """Mean / std over seeds, ignoring NaNs and empty inputs."""
    if not values_per_seed:
        return float("nan"), float("nan")
    a = np.array(values_per_seed, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))
