"""
Feature maps for OOD generalization experiments.

Each FeatureMap encodes a coordinate-system choice (Level-1 commitment):
  transform_x : raw x  → feature matrix fed to regression/deep models
  transform_y : raw y  → target in transformed space
  inverse_y   : transformed prediction → original y space

Experiments import the named constants (IDENTITY, FOURIER, LOGLOG, LOGY)
or build custom maps with fourier_map(freqs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureMap:
    name:        str
    transform_x: Callable[[np.ndarray], np.ndarray]
    transform_y: Callable[[np.ndarray], np.ndarray]
    inverse_y:   Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_log(a: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(a, 1e-30))


def _identity_x(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1)


def _loglog_x(x: np.ndarray) -> np.ndarray:
    return _safe_log(x.ravel()).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def fourier_map(freqs: Sequence[int] = (1, 2), name: str | None = None) -> FeatureMap:
    """Return a Fourier feature map for the given integer frequencies.

    Produces columns [sin(f·x), cos(f·x)] for each f in freqs.
    The standard map for this paper uses freqs=(1, 2): four columns
    {sin x, cos x, sin 2x, cos 2x}, deliberately excluding sin(3x) so
    that OLS cannot represent sin(x)cos(2x) linearly.
    """
    freqs = tuple(freqs)
    _name = name if name is not None else f"Fourier-K{len(freqs)}"

    def _tx(x: np.ndarray) -> np.ndarray:
        xf = x.ravel()
        return np.column_stack([fn(k * xf) for k in freqs for fn in (np.sin, np.cos)])

    return FeatureMap(name=_name, transform_x=_tx,
                      transform_y=lambda y: y, inverse_y=lambda y: y)


# ---------------------------------------------------------------------------
# Standard maps (import these in experiments)
# ---------------------------------------------------------------------------

IDENTITY = FeatureMap(
    name="None",
    transform_x=_identity_x,
    transform_y=lambda y: y,
    inverse_y=lambda y: y,
)

# {sin x, cos x, sin 2x, cos 2x} — the standard Fourier map for this paper.
FOURIER = fourier_map(freqs=(1, 2), name="Fourier")

# Log-log: u = log x as input, v = log y as target.
LOGLOG = FeatureMap(
    name="log-log",
    transform_x=_loglog_x,
    transform_y=_safe_log,
    inverse_y=np.exp,
)

# Log-y: raw x as input, v = log y as target.
LOGY = FeatureMap(
    name="log-y",
    transform_x=_identity_x,
    transform_y=_safe_log,
    inverse_y=np.exp,
)

# Log-periodic: {sin(log x), cos(log x), sin(2 log x), cos(2 log x)}, log y target.
# Used by exp1_3_log_nonlinear.
def _logperiodic_x(x: np.ndarray) -> np.ndarray:
    u = _safe_log(x.ravel())
    return np.column_stack([np.sin(u), np.cos(u), np.sin(2 * u), np.cos(2 * u)])

LOG_PERIODIC = FeatureMap(
    name="log-periodic",
    transform_x=_logperiodic_x,
    transform_y=_safe_log,
    inverse_y=np.exp,
)
