"""Proximity lower-confidence-bound acquisition function for SMAC."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.stats import norm
from smac.acquisition.function import AbstractAcquisitionFunction


class ProximityLowerBoundAcquisition(AbstractAcquisitionFunction):
    """Acquisition on the empirical lower prediction bound of the surrogate (maximised by SMAC).

        bonus(x) = max(gamma(x) * (y_pred(x) - y_lwr(x)), eps * kappa * local_mae(x))
        acq(x)   = -(y_pred(x) - bonus(x))

    ``y_lwr`` and ``local_mae`` come from the surrogate's ``predict_with_intervals`` at confidence
    ``level``; kappa is the matching two-sided normal quantile. gamma(x) is the extractor's density factor
    for the AC/BC variants and 1 otherwise. While the number of observations is at most k (``k_warmup``
    if k is not an int) for the discrete variants A/AC, or zero for the continuous variants B/BC, the
    plain random-forest LCB is used instead.
    """

    def __init__(
        self,
        eps: float = 0.16,
        level: float = 0.95,
        k: int | str = 25,
        k_warmup: int = 25,
        weighting: str = "unweighted",
        use_leaf_weights: bool | None = None,
        weighted: bool = False,
        method: str | None = None,
        fallback_on_low_support: bool = False,
    ) -> None:
        super().__init__()
        self._eps = float(eps)
        self._level = float(level)
        self._k = k
        self._k_warmup = int(k_warmup)
        self._fallback_on_low_support = bool(fallback_on_low_support)
        if weighted is True or use_leaf_weights is True or weighting == "weighted":
            self._weighting = "weighted"
            self._weighted = True
        else:
            self._weighting = "unweighted"
            self._weighted = False
        self._use_leaf_weights = use_leaf_weights
        self._method = method.lower() if isinstance(method, str) else None
        alpha = 1.0 - self._level
        self._kappa = float(norm.ppf(1.0 - alpha / 2.0)) if self._level > 0.0 else 0.0
        self._num_data: int | None = None

    @property
    def name(self) -> str:
        return f"Proximity Lower Bound (level={self._level}, eps={self._eps}, k={self._k})"

    @property
    def meta(self) -> dict[str, Any]:
        meta = super().meta
        meta.update(
            {
                "eps": self._eps,
                "level": self._level,
                "k": self._k,
                "k_warmup": self._k_warmup,
                "kappa": self._kappa,
                "weighting": self._weighting,
                "weighted": self._weighted,
                "method": self._method,
                "density_scaling": self._uses_density_scaling(),
                "fallback_on_low_support": self._fallback_on_low_support,
            }
        )
        return meta

    def _uses_density_scaling(self) -> bool:
        """Whether the bonus is scaled by the density factor gamma(x) (AC and BC variants)."""
        return self._method in (
            "ac",
            "bc",
            "ac_cv",
            "bc_cv",
            "proximity_ac",
            "proximity_bc",
            "proximity_ac_cv",
            "proximity_bc_cv",
        )

    def _is_continuous_method(self) -> bool:
        """True for the continuous variants (B, BC), False for the k-neighbour variants (A, AC).

        Taken from ``method`` if given, otherwise inferred from the model's uncertainty_func, extractor
        class name or ``method`` attribute.
        """
        if self._method is not None:
            m = self._method.lower()
            return m.startswith("proximity_b") or m in (
                "b",
                "bc",
                "b_cv",
                "bc_cv",
                "proximity_b",
                "proximity_bc",
                "proximity_b_cv",
                "proximity_bc_cv",
            )

        if self._model is not None:
            u_func = getattr(self._model, "uncertainty_func", None)
            if isinstance(u_func, str):
                u_str = u_func.lower()
                if u_str.startswith("proximity_b") or u_str in (
                    "b",
                    "bc",
                    "b_cv",
                    "bc_cv",
                    "proximity_b",
                    "proximity_bc",
                    "proximity_b_cv",
                    "proximity_bc_cv",
                ):
                    return True

            extractor = getattr(self._model, "uq_extractor", None) or getattr(self._model, "extractor", None)
            candidates = [extractor, self._model]
            for obj in candidates:
                if obj is not None:
                    cls_name = type(obj).__name__
                    if (
                        cls_name.startswith("ProximityB")
                        or cls_name.startswith("ProximityBC")
                        or "proximityb" in cls_name.lower()
                        or "proximity_b" in cls_name.lower()
                    ):
                        return True
                    if hasattr(obj, "method") and isinstance(obj.method, str):
                        m_obj = obj.method.lower()
                        if m_obj.startswith("proximity_b") or m_obj in (
                            "b",
                            "bc",
                            "b_cv",
                            "bc_cv",
                            "proximity_b",
                            "proximity_bc",
                            "proximity_b_cv",
                            "proximity_bc_cv",
                        ):
                            return True

        return False

    def _update(self, **kwargs: Any) -> None:
        self._num_data = int(kwargs["num_data"]) if "num_data" in kwargs else None

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Surrogate model must be set before computing acquisition values.")

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        N = self._num_data
        if N is None:
            if hasattr(self._model, "last_X") and self._model.last_X is not None:
                N = len(self._model.last_X)
            elif hasattr(self._model, "n_train"):
                N = int(self._model.n_train)
            else:
                N = 0

        is_continuous = self._is_continuous_method()
        if is_continuous:
            should_warmup = N is None or N <= 0
            n_neighbors = "auto"
        else:
            if isinstance(self._k, (int, np.integer)):
                k_eff = int(self._k)
            else:
                k_eff = self._k_warmup
            should_warmup = N is None or N <= k_eff
            n_neighbors = self._k

        # Warm-up: plain random-forest LCB, -(mean - kappa * std).
        if should_warmup:
            if hasattr(self._model, "predict_standard_rf"):
                mean_rf, var_rf = self._model.predict_standard_rf(X)
            else:
                mean_rf, var_rf = self._model.predict_marginalized(X)
            mean_rf = np.asarray(mean_rf, dtype=np.float64).flatten()
            std_rf = np.sqrt(np.maximum(np.asarray(var_rf, dtype=np.float64).flatten(), 1e-10))
            return (-mean_rf + self._kappa * std_rf).reshape(-1, 1)

        # Surrogates differ in which keyword arguments predict_with_intervals accepts; drop them in turn.
        if hasattr(self._model, "predict_with_intervals"):
            try:
                res = self._model.predict_with_intervals(
                    X,
                    n_neighbors=n_neighbors,
                    level=self._level,
                    return_mae=True,
                    weighting=self._weighting,
                    weighted=self._weighted,
                    fallback_on_low_support=self._fallback_on_low_support,
                )
            except TypeError:
                try:
                    res = self._model.predict_with_intervals(
                        X,
                        n_neighbors=n_neighbors,
                        level=self._level,
                        return_mae=True,
                        weighting=self._weighting,
                        fallback_on_low_support=self._fallback_on_low_support,
                    )
                except TypeError:
                    try:
                        res = self._model.predict_with_intervals(
                            X,
                            n_neighbors=n_neighbors,
                            level=self._level,
                            return_mae=True,
                            fallback_on_low_support=self._fallback_on_low_support,
                        )
                    except TypeError:
                        res = self._model.predict_with_intervals(
                            X, n_neighbors=n_neighbors, level=self._level, return_mae=True
                        )
            if len(res) == 4:
                y_pred_lwr, y_pred, _, local_mae = res
            else:
                y_pred_lwr, y_pred, _ = res
                local_mae = getattr(self._model, "oob_mae", 1.0)
        else:
            warnings.warn(
                f"Surrogate model '{type(self._model).__name__}' does not have 'predict_with_intervals'; "
                "falling back to standard Gaussian prediction for Proximity LCB.",
                UserWarning,
                stacklevel=2,
            )
            mean, var = self._model.predict_marginalized(X)
            y_pred = mean.flatten()
            std = np.sqrt(np.maximum(var.flatten(), 1e-10))
            y_pred_lwr = y_pred - self._kappa * std
            local_mae = np.full_like(y_pred, getattr(self._model, "oob_mae", 1.0))

        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        y_pred_lwr = np.asarray(y_pred_lwr, dtype=np.float64).flatten()
        local_mae = np.asarray(local_mae, dtype=np.float64).flatten()
        if local_mae.size == 1 and y_pred.size > 1:
            local_mae = np.full_like(y_pred, local_mae.item())

        scaled = self._uses_density_scaling()
        gamma = (
            np.asarray(self._model.density_factor(X), dtype=np.float64).flatten()
            if scaled and hasattr(self._model, "density_factor")
            else 1.0
        )
        q_bonus = np.maximum(0.0, y_pred - y_pred_lwr)
        delta_floor = self._eps * self._kappa * local_mae if (self._eps > 0.0 and self._kappa > 0.0) else 0.0
        bonus = np.maximum(gamma * q_bonus, delta_floor)
        return (-(y_pred - bonus)).reshape(-1, 1)
