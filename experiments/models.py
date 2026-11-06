"""
Model classes for OOD generalization experiments.

Hierarchy:
  RegressionModel  — supervised, supports_id_error = True
    OLSModel       — ordinary least squares (optionally polynomial)
    SINDyModel     — sparse regression (STLSQ) on a pre-computed feature matrix
    MLPModel       — fully-connected neural network (tanh activations)
    TabPFNModel    — TabPFN zero-shot regression (lazy token load)

  ForecastModel    — zero-shot forecaster, supports_id_error = False
    TimesFMModel   — univariate time-series forecaster (lazy model load)

  nn.Module helpers (torus benchmark):
    LearnedFourierFeatures — per-coordinate trainable Fourier features
                             on continuous inputs, init misspecified

All classes expose a uniform interface so evaluation.py can dispatch on type.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class RegressionModel(ABC):
    supports_id_error: ClassVar[bool] = True

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class ForecastModel(ABC):
    supports_id_error: ClassVar[bool] = False

    @abstractmethod
    def set_context(self, y_ctx: np.ndarray) -> None: ...

    @abstractmethod
    def forecast(self, n_steps: int) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# OLSModel
# ---------------------------------------------------------------------------

class OLSModel(RegressionModel):
    """Ordinary least squares; polynomial expansion when degree > 1 and input is 1-D."""

    def __init__(self, degree: int = 1):
        self.degree = degree
        self._w: np.ndarray | None = None

    def _expand(self, X: np.ndarray) -> np.ndarray:
        if self.degree == 1 or X.shape[1] > 1:
            return np.hstack([X, np.ones((len(X), 1))])
        cols = [X[:, 0] ** d for d in range(1, self.degree + 1)]
        return np.hstack([np.column_stack(cols), np.ones((len(X), 1))])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        A = self._expand(X)
        self._w, _, _, _ = np.linalg.lstsq(A, y.ravel(), rcond=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._expand(X) @ self._w).ravel()


# ---------------------------------------------------------------------------
# MLPModel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SINDyModel
# ---------------------------------------------------------------------------

class SINDyModel(RegressionModel):
    """STLSQ sparse regression on a pre-computed feature matrix.

    Caller is responsible for building the feature matrix (Fourier, poly, etc.).
    threshold controls sparsity: coefficients with |w| <= threshold are zeroed.
    """

    def __init__(self, threshold: float = 0.1, max_iter: int = 20):
        self.threshold = threshold
        self.max_iter = max_iter
        self._w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        w, _, _, _ = np.linalg.lstsq(X, y.ravel(), rcond=None)
        for _ in range(self.max_iter):
            mask = np.abs(w) > self.threshold
            if not mask.any():
                w = np.zeros_like(w)
                break
            ws, _, _, _ = np.linalg.lstsq(X[:, mask], y.ravel(), rcond=None)
            wn = np.zeros_like(w)
            wn[mask] = ws
            if np.array_equal(np.abs(wn) > self.threshold, mask):
                w = wn
                break
            w = wn
        self._w = w

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self._w).ravel()


class MLPModel(RegressionModel):
    """Fully-connected tanh MLP trained with Adam."""

    def __init__(
        self,
        hidden: int = 256,
        layers: int = 5,
        epochs: int = 2000,
        lr: float = 1e-3,
    ):
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self._net = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from utils import MLP, train_mlp, DEVICE
        self._net = MLP(in_dim=X.shape[1], hidden=self.hidden, layers=self.layers)
        self._net = train_mlp(self._net, X, y.ravel(), epochs=self.epochs, lr=self.lr, device=DEVICE)

    def predict(self, X: np.ndarray) -> np.ndarray:
        from utils import evaluate_mlp
        return evaluate_mlp(self._net, X)


# ---------------------------------------------------------------------------
# TabPFN token loading (shared, module-level singleton)
# ---------------------------------------------------------------------------

_TABPFN_TOKEN: str | None = None
_TabPFN_cls = None
_tabpfn_loaded: bool = False


def _load_tabpfn() -> None:
    global _TABPFN_TOKEN, _TabPFN_cls, _tabpfn_loaded
    if _tabpfn_loaded:
        return
    _tabpfn_loaded = True

    _here = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_here, "..", "..", ".env")
    tok = os.environ.get("TABPFN_TOKEN")
    if not tok and os.path.isfile(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TABPFN_TOKEN"):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    _TABPFN_TOKEN = tok

    if _TABPFN_TOKEN:
        try:
            import tabpfn_client
            tabpfn_client.set_access_token(_TABPFN_TOKEN)
            from tabpfn_client import TabPFNRegressor
            _TabPFN_cls = TabPFNRegressor
        except Exception:
            pass


class TabPFNModel(RegressionModel):
    """TabPFN zero-shot regression."""

    def __init__(self):
        _load_tabpfn()
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return _TabPFN_cls is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X_train = X
        self._y_train = y.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if _TabPFN_cls is None:
            return np.full(len(X), float("nan"))
        m = _TabPFN_cls(ignore_pretraining_limits=True)
        m.fit(self._X_train, self._y_train)
        return m.predict(X).flatten()


# ---------------------------------------------------------------------------
# TimesFMModel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LearnedFourierFeatures — torus benchmark misspecified-init baseline
# ---------------------------------------------------------------------------
#
# Adapted from a prior LearnedFourierPE implementation: same
# parametric form (per-coordinate trainable angular frequency, log-space) but
# operating on continuous inputs x ∈ R^{n×d} instead of discrete positions
# torch.arange(seq_len). One log-frequency per torus coordinate.

class LearnedFourierFeatures(nn.Module):
    """Per-coordinate Fourier features with trainable frequencies.

    For input x ∈ R^{n×d}, computes
        φ̂(x) = (sin(ω_1 x_1), cos(ω_1 x_1), …, sin(ω_d x_d), cos(ω_d x_d))
    where each ω_j is a learnable parameter in log-space.

    Init: log_omega is sampled log-uniformly per coordinate in
    [log(init_min), log(init_max)] using a torch.Generator with init_seed
    (so misspecification is reproducible across runs and ω*=1 is in the
    support range — SGD must still locate it). Default range [0.1, 10] spans
    the truth ω*=1 by a factor of 10 in either direction.
    """

    def __init__(
        self,
        d: int,
        init_min: float = 0.1,
        init_max: float = 10.0,
        init_seed: int | None = 0,
    ):
        super().__init__()
        self.d = d
        if init_seed is not None:
            g = torch.Generator().manual_seed(init_seed)
            log_omega = torch.empty(d).uniform_(
                math.log(init_min), math.log(init_max), generator=g,
            )
        else:
            log_omega = torch.empty(d).uniform_(
                math.log(init_min), math.log(init_max),
            )
        self.log_omega = nn.Parameter(log_omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, d)  →  features (n, 2d), order (sin_1, cos_1, …, sin_d, cos_d)
        omega = torch.exp(self.log_omega)         # (d,)
        ang = x * omega.unsqueeze(0)              # (n, d)
        out = torch.empty(x.shape[0], 2 * self.d, device=x.device, dtype=x.dtype)
        out[:, 0::2] = torch.sin(ang)
        out[:, 1::2] = torch.cos(ang)
        return out


class LearnedFourierMLP(nn.Module):
    """End-to-end module: per-coordinate trainable Fourier features → MLP.

    Wraps LearnedFourierFeatures (2d outputs from d inputs) with a tanh MLP
    matching the architecture used elsewhere in the experiments (utils.MLP).
    Frequencies and MLP weights train jointly via Adam.
    """

    def __init__(
        self,
        d: int,
        hidden: int = 256,
        layers: int = 5,
        out_dim: int = 1,
        init_min: float = 0.1,
        init_max: float = 10.0,
        init_seed: int | None = 0,
    ):
        super().__init__()
        self.features = LearnedFourierFeatures(
            d, init_min=init_min, init_max=init_max, init_seed=init_seed,
        )
        seq = [nn.Linear(2 * d, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq.append(nn.Linear(hidden, out_dim))
        self.mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(x))


class TimesFMModel(ForecastModel):
    """Zero-shot univariate forecaster (TimesFM).  Loaded lazily on first use."""

    supports_id_error: ClassVar[bool] = False

    def __init__(
        self,
        context_len: int = 128,
        horizon_len: int = 128,
        backend: str = "cpu",
        checkpoint: str = "google/timesfm-1.0-200m-pytorch",
    ):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.backend = backend
        self.checkpoint = checkpoint
        self._model = None
        self._context: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import timesfm
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=self.context_len,
                horizon_len=self.horizon_len,
                input_patch_len=32,
                output_patch_len=128,
                per_core_batch_size=32,
                backend=self.backend,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.checkpoint,
            ),
        )

    def set_context(self, y_ctx: np.ndarray) -> None:
        self._context = y_ctx.astype(np.float32)

    def forecast(self, n_steps: int) -> np.ndarray:
        self._ensure_loaded()
        pts, _ = self._model.forecast(
            inputs=[self._context],
            freq=[0],
        )
        return pts[0][:n_steps]

    def forecast_with_covariates(
        self,
        y_ctx: np.ndarray,
        dynamic_covariates: dict[str, np.ndarray],
        n_steps: int,
    ) -> np.ndarray:
        """Forecast using the official forecast_with_covariates API.

        dynamic_covariates: {name: array of length context_len + n_steps}
          Each array contains the covariate values for both the context period
          (first context_len points) and the forecast period (next n_steps).
        """
        self._ensure_loaded()
        pts, _ = self._model.forecast_with_covariates(
            inputs=[y_ctx.astype(np.float32)],
            dynamic_numerical_covariates={
                k: [v.astype(np.float32)] for k, v in dynamic_covariates.items()
            },
            freq=[0],
        )
        return pts[0][:n_steps]
