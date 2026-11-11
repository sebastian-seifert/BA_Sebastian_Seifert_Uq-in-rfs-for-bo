"""Proximity A+C: leaf co-occurrence proximity with leaf-density scaling."""

from __future__ import annotations

from dyrf_bo.uq.extractors import UQExtractorRegistry
from dyrf_bo.uq.extractors._proximity import ProximityExtractor


@UQExtractorRegistry.register("proximity_ac")
class ProximityACExtractor(ProximityExtractor):
    """Co-occurrence proximity (no topological decay) with density scaling exponent ``alpha``."""

    def __init__(
        self,
        model,
        device="auto",
        alpha: float = 1.0,
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
        if "decay_lambda" in kwargs or "topological_decay_lambda" in kwargs:
            raise TypeError(
                "ProximityACExtractor uses leaf co-occurrence (P_co) and does not accept 'decay_lambda'. "
                "Use 'proximity_bc' instead."
            )
        self.alpha = kwargs.pop("density_scaling_alpha", alpha)

    def _variant_kwargs(self) -> dict:
        return {
            "use_density_scaling": True,
            "density_scaling_alpha": self.alpha,
            "topological_decay_lambda": None,
        }
