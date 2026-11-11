"""Proximity B+C: topological tree-path decay proximity with walked-density scaling."""

from __future__ import annotations

import numpy as np

from dyrf_bo.uq.extractors import UQExtractorRegistry
from dyrf_bo.uq.extractors._proximity import ProximityExtractor


@UQExtractorRegistry.register("proximity_bc")
class ProximityBCExtractor(ProximityExtractor):
    """Topological decay proximity (``decay_lambda``) with density scaling exponent ``alpha``."""

    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.0,
        alpha=1.0,
        weighting: str = "unweighted_all",
        use_leaf_weights: bool | None = None,
        weighted: bool = False,
        residual_mode: str = "oob",
        cv_folds: int = 5,
        **kwargs,
    ):
        super().__init__(
            model,
            device=device,
            weighting=weighting,
            use_leaf_weights=use_leaf_weights,
            weighted=weighted,
            residual_mode=residual_mode,
            cv_folds=cv_folds,
        )
        self.decay_lambda = decay_lambda
        self.alpha = alpha

    def _variant_kwargs(self) -> dict:
        return {
            "use_density_scaling": True,
            "density_scaling_alpha": self.alpha,
            "topological_decay_lambda": self.decay_lambda,
        }

    def density_factor(self, X: np.ndarray) -> np.ndarray:
        """Density scaling factor gamma(x) >= 1 used by the floored acquisition function."""
        if self.uq_model is None:
            raise RuntimeError("Extractor must be fitted before computing density factor.")
        return self.uq_model.compute_density_factor(X, alpha=self.alpha)


@UQExtractorRegistry.register("proximity_bc_cv")
class ProximityBCCVExtractor(ProximityBCExtractor):
    """Proximity B+C with K-fold cross-validation residuals."""

    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.0,
        alpha=1.0,
        weighting: str = "unweighted_all",
        use_leaf_weights: bool | None = None,
        weighted: bool = False,
        residual_mode: str = "cv",
        cv_folds: int = 5,
        **kwargs,
    ):
        super().__init__(
            model,
            device=device,
            decay_lambda=decay_lambda,
            alpha=alpha,
            weighting=weighting,
            use_leaf_weights=use_leaf_weights,
            weighted=weighted,
            residual_mode=residual_mode,
            cv_folds=cv_folds,
            **kwargs,
        )
