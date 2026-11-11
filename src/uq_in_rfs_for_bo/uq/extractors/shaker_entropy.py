"""Shaker entropy extractor: forest mutual information converted to a standard deviation."""

import numpy as np

from dyrf_bo.uq.entropy import EpistemicQuantifier
from dyrf_bo.uq.extractors import UQExtractorRegistry
from dyrf_bo.uq.extractors.base import BaseEpistemicExtractor


@UQExtractorRegistry.register("shaker_entropy")
class ShakerEntropyExtractor(BaseEpistemicExtractor):
    """Epistemic uncertainty from the Shaker & Hüllermeier entropy decomposition.

    The mutual information ``MI`` (bits) between the prediction and the tree index is mapped to
    target units via the Gaussian identity ``MI = 0.5 * log2(sigma_total^2 / sigma_ale^2)``, i.e.
    ``sigma_epi = sigma_ale * sqrt(4**MI - 1)``. The signal therefore vanishes when the aleatoric
    variance does, whatever the MI.

    Parameters are forwarded to :meth:`EpistemicQuantifier.shaker_get_epistemic_variance`.
    """

    def __init__(
        self,
        model,
        num_samples=10000,
        batch_size="auto",
        backend="auto",
        n_quadrature_points=32,
        method="huber",
        enable_splitting=False,
        **kwargs,
    ):
        super().__init__(model)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.backend = backend
        self.n_quadrature_points = n_quadrature_points
        self.method = method
        self.enable_splitting = enable_splitting
        self.eq = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.eq = EpistemicQuantifier(self.model, X_train, y_train)

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        if self.eq is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
        epistemic_var = self.eq.shaker_get_epistemic_variance(
            X,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            backend=self.backend,
            n_quadrature_points=self.n_quadrature_points,
            method=self.method,
            enable_splitting=self.enable_splitting,
        )
        return np.sqrt(epistemic_var)
