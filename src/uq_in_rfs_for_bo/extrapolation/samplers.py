"""Training and test samplers for the extrapolation study.

Training points come from a Latin hypercube in [-0.5, 0.5]^D; test points are either uniform on
[-1, 1]^D or drawn per distance stratum via Dirichlet combinations and ray casting.
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import LatinHypercube

from .convex_hull import ConvexHullProjectionSolver, ProjectionResult


def sample_subdomain_training(
    n_samples: int,
    dimension: int,
    domain_half_width: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Draw a Latin hypercube sample in the sub-domain [-c, c]^D.

    Parameters
    ----------
    n_samples : int
        Number of training points.
    dimension : int
        Input dimensionality D.
    domain_half_width : float, default=0.5
        Half width c of the sub-domain.
    seed : int or None, default=None
        Seed of the Latin hypercube sampler.

    Returns
    -------
    np.ndarray
        Training points of shape (n_samples, dimension).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if domain_half_width <= 0:
        raise ValueError(f"domain_half_width must be positive, got {domain_half_width}")

    sampler = LatinHypercube(d=dimension, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    return -domain_half_width + 2.0 * domain_half_width * unit_samples


def sample_natural_test(
    n_samples: int,
    dimension: int,
    solver: ConvexHullProjectionSolver,
    seed: int | None = None,
) -> tuple[np.ndarray, ProjectionResult, float]:
    """Draw natural uniform test points on [-1, 1]^D and compute interpolation rate.

    Parameters
    ----------
    n_samples : int
        Number of test points to draw.
    dimension : int
        Feature space dimensionality D.
    solver : ConvexHullProjectionSolver
        Pre-fitted solver for Conv(X_train).
    seed : int | None, default=None
        Random number generator seed.

    Returns
    -------
    tuple[np.ndarray, ProjectionResult, float]
        (X_test, projection_result, p_interp) where p_interp is empirical interpolation rate.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if solver.D != dimension:
        raise ValueError(f"Solver dimension {solver.D} does not match requested dimension {dimension}.")

    rng = np.random.default_rng(seed)
    X_test = rng.uniform(-1.0, 1.0, size=(n_samples, dimension))
    projection_result = solver.project(X_test)
    p_interp = float(np.mean(projection_result.is_interpolating))

    return X_test, projection_result, p_interp


def sample_stratified_test(
    n_per_stratum: int,
    dimension: int,
    X_train: np.ndarray,
    solver: ConvexHullProjectionSolver,
    seed: int | None = None,
) -> tuple[np.ndarray, ProjectionResult, np.ndarray]:
    """Generate test points partitioned into 4 strata via Dirichlet & ray-casting.

    Strata:
    - Stratum 0 (Interpolation): Dirichlet combinations on X_train (alpha = 1_N).
    - Stratum 1 (Near Extrapolation): d_norm in [1e-6, 0.15] via ray-casting.
    - Stratum 2 (Medium Extrapolation): d_norm in (0.15, 0.40] via ray-casting.
    - Stratum 3 (Far Extrapolation): d_norm in (0.40, 0.80] via ray-casting.

    Parameters
    ----------
    n_per_stratum : int
        Number of points per stratum.
    dimension : int
        Feature space dimensionality D.
    X_train : np.ndarray
        Training sample coordinates of shape (N, D).
    solver : ConvexHullProjectionSolver
        Convex hull projection solver.
    seed : int | None, default=None
        Random number generator seed.

    Returns
    -------
    tuple[np.ndarray, ProjectionResult, np.ndarray]
        (X_test, projection_result, strata_labels) where X_test has shape (4 * n_per_stratum, D)
        and strata_labels has shape (4 * n_per_stratum,) with values in {0, 1, 2, 3}.
    """
    if n_per_stratum <= 0:
        raise ValueError(f"n_per_stratum must be positive, got {n_per_stratum}")
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")

    X_train_arr = np.asarray(X_train, dtype=np.float64)
    if X_train_arr.ndim != 2:
        raise ValueError(f"X_train must be 2D array, got {X_train_arr.ndim}D.")
    if X_train_arr.shape[0] == 0:
        raise ValueError("X_train cannot be empty.")
    if X_train_arr.shape[1] != dimension:
        raise ValueError(
            f"X_train dimension {X_train_arr.shape[1]} does not match requested dimension {dimension}."
        )
    if solver.D != dimension:
        raise ValueError(f"Solver dimension {solver.D} does not match requested dimension {dimension}.")

    N_train = X_train_arr.shape[0]
    rng = np.random.default_rng(seed)

    # Stratum 0: random convex combinations of the training points.
    w_dirichlet = rng.dirichlet(np.ones(N_train), size=n_per_stratum)
    X_stratum_0 = w_dirichlet @ X_train_arr

    # Strata 1-3: rays from random training points. Target bounds for the step length differ
    # slightly from the acceptance ranges (stratum 1 must lie strictly outside the hull).
    strata_target_bounds = {
        1: (0.00, 0.15),
        2: (0.15, 0.40),
        3: (0.40, 0.80),
    }
    strata_ranges = {
        1: (1e-6, 0.15),
        2: (0.15, 0.40),
        3: (0.40, 0.80),
    }
    collected: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    sqrt_D = np.sqrt(dimension)
    max_iters = 200
    iters = 0

    while any(len(collected[s]) < n_per_stratum for s in (1, 2, 3)) and iters < max_iters:
        iters += 1
        for s in (1, 2, 3):
            needed = n_per_stratum - len(collected[s])
            if needed <= 0:
                continue

            t_low, t_high = strata_target_bounds[s]
            batch_size = max(needed * 4, 100)

            # Random anchor x0 in X_train
            idx = rng.integers(0, N_train, size=batch_size)
            x0 = X_train_arr[idx]

            # Isotropic Gaussian unit ray v = z / ||z||_2
            z = rng.standard_normal((batch_size, dimension))
            norm_z = np.linalg.norm(z, axis=1, keepdims=True)
            norm_z = np.maximum(norm_z, 1e-12)
            v = z / norm_z

            # Step along the ray by a target distance and clip to the domain [-1, 1]^D.
            d_target = rng.uniform(t_low, t_high, size=batch_size)
            cand = x0 + d_target[:, None] * sqrt_D * v
            cand = np.clip(cand, -1.0, 1.0)

            d_norm = solver.distance(cand)

            # The clipped candidate's true distance decides its stratum.
            for target_s, (low, high) in strata_ranges.items():
                if len(collected[target_s]) >= n_per_stratum:
                    continue
                mask = (d_norm > low) & (d_norm <= high)

                matches = cand[mask]
                if len(matches) > 0:
                    avail = n_per_stratum - len(collected[target_s])
                    collected[target_s].extend(matches[:avail])

    for s in (1, 2, 3):
        if len(collected[s]) < n_per_stratum:
            raise RuntimeError(
                f"Could not accumulate {n_per_stratum} points for stratum {s} "
                f"(collected {len(collected[s])}) after {max_iters} iterations."
            )

    X_stratum_1 = np.asarray(collected[1][:n_per_stratum], dtype=np.float64)
    X_stratum_2 = np.asarray(collected[2][:n_per_stratum], dtype=np.float64)
    X_stratum_3 = np.asarray(collected[3][:n_per_stratum], dtype=np.float64)

    X_test = np.vstack([X_stratum_0, X_stratum_1, X_stratum_2, X_stratum_3])
    strata_labels = np.concatenate(
        [
            np.zeros(n_per_stratum, dtype=np.int64),
            np.ones(n_per_stratum, dtype=np.int64),
            np.full(n_per_stratum, 2, dtype=np.int64),
            np.full(n_per_stratum, 3, dtype=np.int64),
        ]
    )

    projection_result = solver.project(X_test)
    return X_test, projection_result, strata_labels
