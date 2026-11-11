"""Common implementation of the extractors backed by :class:`GPUProximityRegressionUQ`."""

from __future__ import annotations

import numpy as np

from dyrf_bo.uq.extractors.base import BaseEpistemicExtractor
from dyrf_bo.uq.proximity import GPUProximityRegressionUQ


class ProximityExtractor(BaseEpistemicExtractor):
    """Wraps a :class:`GPUProximityRegressionUQ` engine built on the surrogate forest.

    Subclasses choose the proximity variant through :meth:`_variant_kwargs`.
    """

    def __init__(
        self,
        model,
        device="auto",
        weighting: str = "unweighted_all",
        use_leaf_weights: bool | None = None,
        weighted: bool = False,
        residual_mode: str = "oob",
        cv_folds: int = 5,
    ):
        super().__init__(model)
        self.device = device
        self.weighting = weighting
        self.use_leaf_weights = use_leaf_weights
        self.weighted = weighted
        self.residual_mode = residual_mode
        self.cv_folds = cv_folds
        self.uq_model = None

    def _variant_kwargs(self) -> dict:
        """Proximity-variant arguments passed to :class:`GPUProximityRegressionUQ`."""
        raise NotImplementedError

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.uq_model = GPUProximityRegressionUQ(
            self.model,
            X_train,
            y_train,
            device=self.device,
            weighting=self.weighting,
            use_leaf_weights=self.use_leaf_weights,
            weighted=self.weighted,
            residual_mode=self.residual_mode,
            cv_folds=self.cv_folds,
            **self._variant_kwargs(),
        )
        self.uq_model.fit()

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        return self.uq_model.compute_uq(X)

    @property
    def oob_mae(self) -> float:
        """Mean absolute OOB (or CV) residual of the fitted engine, 1.0 before fitting."""
        if self.uq_model is not None and hasattr(self.uq_model, "oob_mae"):
            return self.uq_model.oob_mae
        return 1.0

    def predict_with_intervals(
        self,
        X: np.ndarray,
        n_neighbors: int | str = "auto",
        level: float = 0.95,
        return_mae: bool = False,
        weighting: str | None = None,
        use_leaf_weights: bool | None = None,
        weighted: bool | None = None,
        **kwargs,
    ):
        """Delegate to :meth:`GPUProximityRegressionUQ.predict_with_intervals`.

        Weighting arguments left as ``None`` fall back to the values given at construction.
        """
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before computing intervals.")
        return self.uq_model.predict_with_intervals(
            X,
            n_neighbors=n_neighbors,
            level=level,
            return_mae=return_mae,
            weighting=weighting if weighting is not None else self.weighting,
            use_leaf_weights=use_leaf_weights if use_leaf_weights is not None else self.use_leaf_weights,
            weighted=weighted if weighted is not None else self.weighted,
            **kwargs,
        )
