"""SMAC random forest surrogate whose variance is replaced by a pluggable uncertainty signal."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import norm
from smac.model.random_forest import RandomForest

from dyrf_bo.uq.extractors import UQExtractorRegistry


class CustomUncertaintyRandomForest(RandomForest):
    """SMAC ``RandomForest`` that predicts (mean, U(x)^2) instead of the tree variance.

    ``uncertainty_func`` is either a ``UQExtractorRegistry`` key, whose extractor is fitted on the trained
    forest (``self._rf``), or a callable ``f(forest, X, y_train) -> U(X)``. With fewer than
    ``min_samples_uq`` observations the plain SMAC prediction is used.
    """

    def __init__(
        self,
        uncertainty_func: str | Callable[[Any, np.ndarray, np.ndarray], np.ndarray] = "standard_disagreement",
        oob_score: bool = True,
        extractor_kwargs: dict | None = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        ratio_features: float = 1.0,
        log_y: bool = True,
        min_samples_uq: int = 5,
        **kwargs,
    ):
        super().__init__(
            oob_score=oob_score,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            ratio_features=ratio_features,
            log_y=log_y,
            **kwargs,
        )
        self.uncertainty_func = uncertainty_func
        self.extractor_kwargs = extractor_kwargs
        self.min_samples_uq = min_samples_uq
        self.uq_extractor = None
        self.last_X = None
        self.last_y = None

    @property
    def extractor(self):
        """Alias for ``uq_extractor``."""
        return self.uq_extractor

    def train(
        self, X: np.ndarray, y: np.ndarray | None = None, Y: np.ndarray | None = None, **kwargs
    ) -> CustomUncertaintyRandomForest:
        """Train the forest and, once enough data is available, the uncertainty extractor."""
        y_target = Y if y is None else y
        if y_target is None:
            raise ValueError("Target values must be provided via 'y' or 'Y'.")
        super().train(X, y_target)
        X_clean = self._impute_inactive(X)
        self.last_X = X_clean
        self.last_y = y_target.flatten()

        if len(X_clean) < self.min_samples_uq:
            self.uq_extractor = None
            return self

        # self._rf is SMAC's EPMRandomForest (an sklearn RandomForestRegressor); the extractor reuses it.
        if isinstance(self.uncertainty_func, str) and self._rf is not None:
            self.uq_extractor = UQExtractorRegistry.get(
                self.uncertainty_func, self._rf, **(self.extractor_kwargs or {})
            )
            self.uq_extractor.fit(X_clean, self.last_y)
        return self

    def _predict(
        self, X: np.ndarray, covariance_type: str | None = "diagonal"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, U(X)^2), so SMAC's acquisition functions use U(X) as standard deviation."""
        if (self.last_X is not None and len(self.last_X) < self.min_samples_uq) or (
            isinstance(self.uncertainty_func, str) and self.uq_extractor is None
        ):
            return super()._predict(X, covariance_type=covariance_type)

        mean, _ = super()._predict(X, covariance_type=covariance_type)
        X_clean = self._impute_inactive(X)

        if isinstance(self.uncertainty_func, str):
            unc_signal = self.uq_extractor.extract_epistemic_signal(X_clean)
        elif callable(self.uncertainty_func):
            unc_signal = self.uncertainty_func(self._rf or self, X_clean, self.last_y)
        else:
            raise ValueError(f"Unsupported uncertainty_func: {self.uncertainty_func}")

        var = (unc_signal**2).reshape(-1, 1)
        var = np.maximum(var, 1e-10)
        return mean, var

    def predict_standard_rf(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plain SMAC random forest (mean, variance), ignoring the uncertainty extractor."""
        X_clean = self._impute_inactive(X)
        return super()._predict(X_clean)

    @property
    def oob_mae(self) -> float:
        """Out-of-bag MAE from the extractor if it provides one, else from the forest (1.0 if unavailable)."""
        if self.uq_extractor is not None and hasattr(self.uq_extractor, "oob_mae"):
            return self.uq_extractor.oob_mae
        if hasattr(self._rf, "oob_prediction_") and self.last_y is not None:
            try:
                valid_mask = ~np.isnan(self._rf.oob_prediction_)
                if np.any(valid_mask):
                    return float(
                        np.mean(np.abs(self.last_y[valid_mask] - self._rf.oob_prediction_[valid_mask]))
                    )
            except Exception:
                pass
        return 1.0

    def _ensure_extractor(self) -> None:
        """Fit the extractor now if training skipped it but enough data is available."""
        if (
            self.uq_extractor is None
            and self._rf is not None
            and isinstance(self.uncertainty_func, str)
            and (self.last_X is not None and len(self.last_X) >= self.min_samples_uq)
        ):
            self.uq_extractor = UQExtractorRegistry.get(
                self.uncertainty_func, self._rf, **(self.extractor_kwargs or {})
            )
            self.uq_extractor.fit(self.last_X, self.last_y)

    def density_factor(self, X: np.ndarray) -> np.ndarray:
        """Density scaling factor gamma(x) of the extractor, or ones if it has none."""
        X_clean = self._impute_inactive(X)
        self._ensure_extractor()

        if self.uq_extractor is not None and hasattr(self.uq_extractor, "density_factor"):
            return np.asarray(self.uq_extractor.density_factor(X_clean), dtype=np.float64).flatten()
        return np.ones(len(X_clean), dtype=np.float64)

    def predict_with_intervals(
        self,
        X: np.ndarray,
        n_neighbors: int | str = "auto",
        level: float = 0.95,
        return_mae: bool = False,
        **kwargs,
    ):
        """(lower, mean, upper[, mae]) prediction intervals from the extractor.

        Falls back to Gaussian intervals around ``_predict`` (with the global OOB MAE) if the extractor
        does not implement ``predict_with_intervals``.
        """
        X_clean = self._impute_inactive(X)
        self._ensure_extractor()

        if self.uq_extractor is not None and hasattr(self.uq_extractor, "predict_with_intervals"):
            return self.uq_extractor.predict_with_intervals(
                X_clean, n_neighbors=n_neighbors, level=level, return_mae=return_mae, **kwargs
            )

        if self.uq_extractor is not None:
            extractor_name = type(self.uq_extractor).__name__
            warnings.warn(
                f"Active UQ extractor '{extractor_name}' does not implement 'predict_with_intervals'; "
                "falling back to Gaussian prediction intervals.",
                UserWarning,
                stacklevel=2,
            )
        mean, var = self._predict(X_clean)
        mean = mean.flatten()
        std = np.sqrt(var.flatten())
        alpha = 1.0 - level
        kappa = float(norm.ppf(1.0 - alpha / 2.0)) if level > 0.0 else 0.0
        y_lwr = mean - kappa * std
        y_upr = mean + kappa * std
        if return_mae:
            mae = np.full_like(mean, self.oob_mae)
            return y_lwr, mean, y_upr, mae
        return y_lwr, mean, y_upr
