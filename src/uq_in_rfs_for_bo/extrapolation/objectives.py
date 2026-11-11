"""Vectorized benchmark objectives (sphere, Rosenbrock, Rastrigin, Ackley) on [-1, 1]^D.

``StandardizedObjective`` z-scores outputs with the mean and std of the training targets.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere synthetic benchmark function.

    Formula: f(x) = sum_{j=1}^D x_j^2.
    Global minimum at x = 0 with f(0) = 0.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (..., D).

    Returns
    -------
    np.ndarray
        Evaluated function values of shape (...).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension (D,).")
    if x_arr.shape[-1] == 0:
        raise ValueError("Feature dimension D cannot be 0.")
    return np.sum(x_arr**2, axis=-1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock synthetic benchmark function (ill-conditioned valley).

    Formula: f(x) = sum_{j=1}^{D-1} [100 (x_{j+1} - x_j^2)^2 + (1 - x_j)^2].
    Global minimum at x = (1, ..., 1) with f(1) = 0. Requires D >= 2.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (..., D).

    Returns
    -------
    np.ndarray
        Evaluated function values of shape (...).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension (D,).")
    D = x_arr.shape[-1]
    if D < 2:
        raise ValueError(f"Rosenbrock function requires dimension D >= 2, got {D}.")
    return np.sum(
        100.0 * (x_arr[..., 1:] - x_arr[..., :-1] ** 2) ** 2 + (1.0 - x_arr[..., :-1]) ** 2,
        axis=-1,
    )


def rastrigin(x: np.ndarray) -> np.ndarray:
    """Rastrigin synthetic benchmark function (highly multimodal).

    Formula: f(x) = 10 D + sum_{j=1}^D [x_j^2 - 10 cos(2 pi x_j)].
    Global minimum at x = 0 with f(0) = 0.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (..., D).

    Returns
    -------
    np.ndarray
        Evaluated function values of shape (...).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension (D,).")
    D = x_arr.shape[-1]
    if D == 0:
        raise ValueError("Feature dimension D cannot be 0.")
    return 10.0 * D + np.sum(x_arr**2 - 10.0 * np.cos(2.0 * np.pi * x_arr), axis=-1)


def ackley(x: np.ndarray) -> np.ndarray:
    """Ackley synthetic benchmark function (outer plateau with central basin).

    Formula: f(x) = -20 exp(-0.2 sqrt(1/D sum x_j^2)) - exp(1/D sum cos(2 pi x_j)) + 20 + e.
    Global minimum at x = 0 with f(0) = 0.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (..., D).

    Returns
    -------
    np.ndarray
        Evaluated function values of shape (...).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension (D,).")
    D = x_arr.shape[-1]
    if D == 0:
        raise ValueError("Feature dimension D cannot be 0.")
    sum_sq = np.sum(x_arr**2, axis=-1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x_arr), axis=-1)
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / D)) - np.exp(sum_cos / D) + 20.0 + np.e


class StandardizedObjective:
    """Standardized wrapper for benchmark objective functions.

    Evaluates base function on X_train, computes training sample mean and std,
    and standardizes outputs: y_tilde = (y - mean_train) / std_train.

    Parameters
    ----------
    base_func : Callable[[np.ndarray], np.ndarray]
        Vectorized base function accepting array of shape (..., D).
    bounds : tuple[float, float] | np.ndarray, default=(-1.0, 1.0)
        Domain bounds for search space.
    name : str | None, default=None
        Identifier name of objective function.
    """

    def __init__(
        self,
        base_func: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float] | np.ndarray = (-1.0, 1.0),
        name: str | None = None,
    ) -> None:
        self.base_func = base_func
        self.bounds = bounds
        self.name = name or getattr(base_func, "__name__", "objective")
        self.mean_train: float | None = None
        self.std_train: float | None = None
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> StandardizedObjective:
        """Compute training mean and standard deviation from X_train.

        Parameters
        ----------
        X_train : np.ndarray
            Training coordinates of shape (N, D).

        Returns
        -------
        StandardizedObjective
            Self reference for fluent chaining.
        """
        y_train = np.asarray(self.base_func(X_train), dtype=np.float64)
        self.mean_train = float(np.mean(y_train))
        raw_std = float(np.std(y_train, ddof=1))
        self.std_train = max(raw_std, 1e-8)
        self.is_fitted = True
        return self

    def evaluate(self, X: np.ndarray, standardized: bool = True) -> np.ndarray:
        """Evaluate objective on input coordinates.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (..., D).
        standardized : bool, default=True
            Whether to return z-score standardized values (y - mean) / std.

        Returns
        -------
        np.ndarray
            Evaluated values of shape (...).
        """
        y = np.asarray(self.base_func(X), dtype=np.float64)
        if standardized:
            if not self.is_fitted:
                raise RuntimeError(
                    f"Objective '{self.name}' must be fit before evaluating with standardized=True."
                )
            return (y - self.mean_train) / self.std_train
        return y

    def __call__(self, X: np.ndarray, standardized: bool = True) -> np.ndarray:
        """Alias for evaluate."""
        return self.evaluate(X, standardized=standardized)


OBJECTIVE_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}


def get_objective(
    name: str,
    bounds: tuple[float, float] | np.ndarray = (-1.0, 1.0),
) -> StandardizedObjective:
    """Retrieve a StandardizedObjective by name from the registry.

    Parameters
    ----------
    name : str
        Name of benchmark function ('sphere', 'rosenbrock', 'rastrigin', 'ackley').
    bounds : tuple[float, float] | np.ndarray, default=(-1.0, 1.0)
        Domain bounds.

    Returns
    -------
    StandardizedObjective
        Configured standardized objective instance.
    """
    key = name.lower()
    if key not in OBJECTIVE_REGISTRY:
        available = list(OBJECTIVE_REGISTRY.keys())
        raise KeyError(f"Objective '{name}' not found. Available objectives: {available}")
    return StandardizedObjective(base_func=OBJECTIVE_REGISTRY[key], bounds=bounds, name=key)
