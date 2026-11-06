"""
Evaluation protocol for OOD generalization experiments.

Provides:
  EvalResult           — typed container for (id_error, ood_error)
  evaluate()           — dispatches on model type; enforces ID-null for ForecastModel
  fmt_error()          — renders float | None → string ("--" for null)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_maps import FeatureMap
from models import RegressionModel, ForecastModel


@dataclass
class EvalResult:
    id_error:  float | None   # None → JSON null → "--" in tables
    ood_error: float

    def to_dict(self) -> dict:
        return {"id_error": self.id_error, "ood_error": self.ood_error}


def fmt_error(e: float | None, decimals: int = 1) -> str:
    return "--" if e is None else f"{e:.{decimals}f}"


def evaluate(
    model,
    feature_map: FeatureMap,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    x_id: np.ndarray | None = None,
    y_id: np.ndarray | None = None,
) -> EvalResult:
    """Fit (or configure) model and return ID and OOD errors.

    For RegressionModel:
      - transforms x/y with feature_map, fits, evaluates on x_ood (and x_id if given)
    For ForecastModel:
      - transforms y_train with feature_map.transform_y, calls set_context + forecast
      - id_error is always None (single source of truth: ForecastModel.supports_id_error=False)
    """
    from utils import ood_relative_error

    if isinstance(model, ForecastModel):
        y_ctx = feature_map.transform_y(y_train)
        model.set_context(y_ctx)
        n_ood = len(y_ood)
        y_fore_t = model.forecast(n_ood)
        y_pred = feature_map.inverse_y(y_fore_t)
        return EvalResult(id_error=None, ood_error=ood_relative_error(y_pred, y_ood))

    elif isinstance(model, RegressionModel):
        X_train = feature_map.transform_x(x_train)
        y_train_t = feature_map.transform_y(y_train)
        model.fit(X_train, y_train_t)

        X_ood = feature_map.transform_x(x_ood)
        y_pred_ood_t = model.predict(X_ood)
        y_pred_ood = feature_map.inverse_y(y_pred_ood_t)
        ood_err = ood_relative_error(y_pred_ood, y_ood)

        id_err = None
        if x_id is not None and y_id is not None:
            X_id = feature_map.transform_x(x_id)
            y_pred_id_t = model.predict(X_id)
            y_pred_id = feature_map.inverse_y(y_pred_id_t)
            id_err = ood_relative_error(y_pred_id, y_id)

        return EvalResult(id_error=id_err, ood_error=ood_err)

    else:
        raise TypeError(f"Unknown model type: {type(model)}")
