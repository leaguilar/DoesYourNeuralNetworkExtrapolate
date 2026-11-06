"""Shared per-condition aggregators used by the main-paper excerpt builders."""

import math
import sys
from pathlib import Path

import numpy as np

# Make tables/_fmt importable as a top-level module when this file is loaded
# either as a script (cell mode) or as a package member.
_HERE = Path(__file__).resolve().parent
_TABLES = _HERE.parent
for _p in (_HERE, _TABLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _fmt import fmt


def cond_metric(runs, cond_key, metric="ood_error"):
    """Mean ± std of `metric` across `runs` for a single condition. None if missing.

    Looks up `cond_key` first under the run's `conditions` dict, then under
    `results` (Kepler / RW1 schema).
    """
    vals = []
    for r in runs:
        c = r.get("conditions", {}).get(cond_key)
        if c is None:
            c = r.get("results", {}).get(cond_key)
        if c is None:
            continue
        v = c.get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def fmt_or_dash(t, bold=False, decimals=1):
    return "--" if t is None else fmt(t[0], t[1], bold=bold, decimals=decimals)


def dgp_metric(runs, dgp, cond, metric):
    """Mean ± std of `metric` for runs[*]['dgps'][dgp][cond][metric]. None if missing."""
    vals = []
    for r in runs:
        try:
            v = r["dgps"][dgp][cond][metric]
        except (KeyError, TypeError):
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))
