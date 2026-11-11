"""Proximity B: topological tree-path decay proximity without density scaling."""

from __future__ import annotations

from dyrf_bo.uq.extractors import UQExtractorRegistry
from dyrf_bo.uq.extractors._proximity import ProximityExtractor


@UQExtractorRegistry.register("proximity_b")
class ProximityBExtractor(ProximityExtractor):
    """Proximity ``exp(-decay_lambda * d)`` over tree-path distances ``d``; OOB residuals."""

    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.345,
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

    def _variant_kwargs(self) -> dict:
        return {"use_density_scaling": False, "topological_decay_lambda": self.decay_lambda}


@UQExtractorRegistry.register("proximity_b_cv")
class ProximityBCVExtractor(ProximityBExtractor):
    """Proximity B with K-fold cross-validation residuals."""

    def __init__(
        self,
        model,
        device="auto",
        decay_lambda=1.345,
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
            weighting=weighting,
            use_leaf_weights=use_leaf_weights,
            weighted=weighted,
            residual_mode=residual_mode,
            cv_folds=cv_folds,
            **kwargs,
        )
