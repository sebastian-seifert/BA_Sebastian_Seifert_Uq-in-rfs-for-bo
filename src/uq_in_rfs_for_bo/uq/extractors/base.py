"""Abstract interface shared by all epistemic-uncertainty extractors."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEpistemicExtractor(ABC):
    """Base class for extractors that turn a fitted random forest into an uncertainty signal.

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted random forest surrogate.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Precompute the training statistics needed by :meth:`extract_epistemic_signal`.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
        y_train : np.ndarray of shape (n_samples,)
        """

    @abstractmethod
    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        """Return the epistemic uncertainty for each row of ``X``, shape (n_samples,)."""
