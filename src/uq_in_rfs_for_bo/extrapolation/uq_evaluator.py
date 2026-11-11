"""Evaluation of ten random-forest uncertainty estimates for the extrapolation study.

Non-proximity methods: Hutter total variance and Shaker epistemic (Huber entropy). Proximity
methods: A, B, AC and BC, each with an OOB and a 5-fold CV residual pool.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from dyrf_bo.uq.entropy import EpistemicQuantifier
from dyrf_bo.uq.proximity import GPUProximityRegressionUQ

APPROACH_NAMES: list[str] = [
    "hutter_total",
    "shaker_epistemic",
    "prox_a_oob",
    "prox_a_cv",
    "prox_b_oob",
    "prox_b_cv",
    "prox_ac_oob",
    "prox_ac_cv",
    "prox_bc_oob",
    "prox_bc_cv",
]

NON_PROXIMITY_APPROACHES: list[str] = [
    "hutter_total",
    "shaker_epistemic",
]

PROXIMITY_APPROACHES: list[str] = [
    "prox_a_oob",
    "prox_a_cv",
    "prox_b_oob",
    "prox_b_cv",
    "prox_ac_oob",
    "prox_ac_cv",
    "prox_bc_oob",
    "prox_bc_cv",
]

COVERAGE_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
COVERAGE_SUFFIXES: dict[float, str] = {0.50: "50", 0.80: "80", 0.90: "90", 0.95: "95"}

# Upper normal quantile for each central interval level.
_GAUSSIAN_QUANTILES: dict[str, float] = {"50": 0.75, "80": 0.90, "90": 0.95, "95": 0.975}

# (suffix, lower quantile, upper quantile) of the residual distribution.
_RESIDUAL_QUANTILES: list[tuple[str, float, float]] = [
    ("50", 0.25, 0.75),
    ("80", 0.10, 0.90),
    ("90", 0.05, 0.95),
    ("95", 0.025, 0.975),
]

SURROGATE_CONFIGS: dict[str, dict[str, Any]] = {
    "smac_default": {
        "n_estimators": 10,
        "max_depth": 2**20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "mature": {
        "n_estimators": 100,
        "max_depth": 2**20,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "shallow": {
        "n_estimators": 25,
        "max_depth": 4,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "coarse": {
        "n_estimators": 25,
        "max_depth": 2**20,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
}


def create_surrogate_rf(
    surrogate_type: str = "smac_default",
    seed: int | None = None,
    **kwargs: Any,
) -> RandomForestRegressor:
    """Create an unfitted random forest for one of the surrogate configurations.

    Parameters
    ----------
    surrogate_type : str, default="smac_default"
        Key into ``SURROGATE_CONFIGS``.
    seed : int or None, default=None
        ``random_state`` of the forest.
    **kwargs
        Overrides for the configuration's hyperparameters.

    Returns
    -------
    RandomForestRegressor
    """
    if surrogate_type not in SURROGATE_CONFIGS:
        raise ValueError(
            f"Unknown surrogate_type '{surrogate_type}'. "
            f"Supported surrogates: {list(SURROGATE_CONFIGS.keys())}"
        )
    params = dict(SURROGATE_CONFIGS[surrogate_type])
    params["random_state"] = seed
    params.update(kwargs)
    return RandomForestRegressor(**params)


def _gaussian_intervals(app: str, y_hat: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    """Symmetric Gaussian prediction intervals ``y_hat +- z * sigma`` at all coverage levels."""
    out: dict[str, np.ndarray] = {}
    for sfx, q in _GAUSSIAN_QUANTILES.items():
        z = float(norm.ppf(q))
        out[f"int_lwr_{app}_{sfx}"] = (y_hat - z * sigma).astype(np.float64)
        out[f"int_upr_{app}_{sfx}"] = (y_hat + z * sigma).astype(np.float64)
    return out


class MultiUQResult:
    """Predictions, uncertainties and prediction intervals for all approaches.

    For every approach ``app`` in ``APPROACH_NAMES`` the attributes ``u_{app}`` (exploration
    half-width), ``sigma_{app}``, ``int_lwr_{app}_{sfx}`` / ``int_upr_{app}_{sfx}`` for
    ``sfx`` in ``{"50", "80", "90", "95"}`` and the un-suffixed 95 % bounds ``int_lwr_{app}`` /
    ``int_upr_{app}`` are set. Values can also be read mapping-style, e.g. ``res["u_prox_a_oob"]``;
    ``keys()`` additionally lists legacy alias names and the available diagnostics.
    """

    _DIAGNOSTICS = (
        "var_between",
        "var_within",
        "var_total",
        "delta_floor",
        "local_mae",
        "q_lower",
        "u_hutter_between",
        "u_hutter_within",
        "u_shaker_total",
        "shaker_mi",
        "shaker_total_entropy",
        "u_rf_fire_lower",
    )

    def __init__(self, y_hat: np.ndarray, **kwargs: Any) -> None:
        self.y_hat = np.asarray(y_hat)
        for name in self._DIAGNOSTICS:
            setattr(self, name, kwargs.get(name))

        for app in APPROACH_NAMES:
            setattr(self, f"u_{app}", np.asarray(kwargs[f"u_{app}"]))
            setattr(self, f"sigma_{app}", np.asarray(kwargs[f"sigma_{app}"]))
            for sfx in ("50", "80", "90", "95"):
                setattr(self, f"int_lwr_{app}_{sfx}", np.asarray(kwargs[f"int_lwr_{app}_{sfx}"]))
                setattr(self, f"int_upr_{app}_{sfx}", np.asarray(kwargs[f"int_upr_{app}_{sfx}"]))
            setattr(self, f"int_lwr_{app}", getattr(self, f"int_lwr_{app}_95"))
            setattr(self, f"int_upr_{app}", getattr(self, f"int_upr_{app}_95"))

    # Legacy names, still written to the per-point parquet and summary JSON.
    @property
    def u_slcb(self) -> np.ndarray:
        return self.u_hutter_total

    @property
    def u_prox_a_lower(self) -> np.ndarray:
        return self.u_prox_a_oob

    @property
    def u_prox_b_lower(self) -> np.ndarray:
        return self.u_prox_b_oob

    @property
    def u_prox_ac_lower(self) -> np.ndarray:
        return self.u_prox_ac_oob

    @property
    def u_prox_bc_lower(self) -> np.ndarray:
        return self.u_prox_bc_oob

    @property
    def u_plcb(self) -> np.ndarray:
        return self.u_prox_a_oob

    @property
    def u_plcb_lower(self) -> np.ndarray:
        return self.u_prox_a_oob

    @property
    def u_rf_fire_half(self) -> np.ndarray:
        return (self.int_upr_prox_a_oob - self.int_lwr_prox_a_oob) / 2.0

    @property
    def u_prox_a_half(self) -> np.ndarray:
        return (self.int_upr_prox_a_oob - self.int_lwr_prox_a_oob) / 2.0

    @property
    def u_prox_b_half(self) -> np.ndarray:
        return (self.int_upr_prox_b_oob - self.int_lwr_prox_b_oob) / 2.0

    @property
    def u_prox_bc_half(self) -> np.ndarray:
        return (self.int_upr_prox_bc_oob - self.int_lwr_prox_bc_oob) / 2.0

    @property
    def u_plcb_half(self) -> np.ndarray:
        return (self.int_upr_prox_a_oob - self.int_lwr_prox_a_oob) / 2.0

    def __getitem__(self, key: str) -> np.ndarray:
        if hasattr(self, key) and not key.startswith("_"):
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                return val
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}.")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key) and not key.startswith("_")

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> tuple[str, ...]:
        k_list = ["y_hat"]
        for app in APPROACH_NAMES:
            k_list.append(f"u_{app}")
            k_list.append(f"sigma_{app}")
            k_list.append(f"int_lwr_{app}")
            k_list.append(f"int_upr_{app}")
            for sfx in ("50", "80", "90", "95"):
                k_list.append(f"int_lwr_{app}_{sfx}")
                k_list.append(f"int_upr_{app}_{sfx}")

        k_list.extend(
            [
                "u_slcb",
                "u_plcb",
                "u_prox_a_lower",
                "u_prox_b_lower",
                "u_prox_ac_lower",
                "u_prox_bc_lower",
                "u_plcb_lower",
                "u_rf_fire_half",
                "u_prox_a_half",
                "u_prox_b_half",
                "u_prox_bc_half",
                "u_plcb_half",
            ]
        )

        diag_keys = (
            "var_between",
            "var_within",
            "var_total",
            "q_lower",
            "delta_floor",
            "local_mae",
            "u_hutter_between",
            "u_hutter_within",
            "u_shaker_total",
            "shaker_mi",
            "shaker_total_entropy",
            "u_rf_fire_lower",
        )
        for dk in diag_keys:
            if getattr(self, dk, None) is not None:
                k_list.append(dk)
        return tuple(k_list)

    def values(self) -> tuple[np.ndarray, ...]:
        return tuple(getattr(self, k) for k in self.keys())

    def items(self) -> tuple[tuple[str, np.ndarray], ...]:
        return tuple((k, getattr(self, k)) for k in self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return getattr(self, key)
        return default

    def to_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.keys()}


class MultiUQEvaluator:
    """Fit a random forest and evaluate all ten uncertainty approaches on query points.

    Parameters
    ----------
    seed : int or None, default=None
        Seed of the forest and of the CV fold split.
    n_trees : int, default=10
        Number of trees. The value 10 means "use the surrogate configuration's default".
    k : int, default=28
        Neighbourhood size of proximity method A.
    epsilon : float, default=0.080791
        Exploration floor coefficient (only used for the ``delta_floor`` diagnostic).
    topological_decay_lambda : float, default=0.20486
        Decay rate of the tree-path proximity ``exp(-lambda * d_t)``.
    density_scaling_alpha : float, default=1.0
        Exponent of the density multiplier ``gamma`` used by AC and BC.
    kappa : float, default=1.96
        Multiplier from standard deviation to exploration half-width.
    device : str, default="cpu"
        Backend for the proximity and entropy engines.
    batch_size : int or str, default=512
        Query chunk size; a string selects 512.
    surrogate_type : str, default="smac_default"
        Key into ``SURROGATE_CONFIGS``.
    eval_mode : {"all", "unweighted_proximity_only"}, default="all"
        ``"unweighted_proximity_only"`` skips the Shaker entropy computation and returns zeros
        for it.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_trees: int = 10,
        k: int = 28,
        epsilon: float = 0.080791,
        topological_decay_lambda: float = 0.20486,
        density_scaling_alpha: float = 1.0,
        kappa: float = 1.96,
        device: str = "cpu",
        batch_size: int | str = 512,
        surrogate_type: str = "smac_default",
        eval_mode: str = "all",
    ) -> None:
        if eval_mode not in ("all", "unweighted_proximity_only"):
            raise ValueError(
                f"Unknown eval_mode '{eval_mode}'. Must be 'all' or 'unweighted_proximity_only'."
            )
        self.eval_mode = eval_mode
        self.surrogate_type = surrogate_type
        self.seed = seed

        if n_trees != 10:
            resolved_n_trees = n_trees
        else:
            resolved_n_trees = SURROGATE_CONFIGS.get(surrogate_type, {}).get("n_estimators", n_trees)

        self.n_trees = resolved_n_trees
        self.k = k
        self.epsilon = epsilon
        self.topological_decay_lambda = topological_decay_lambda
        self.density_scaling_alpha = density_scaling_alpha
        self.kappa = kappa
        self.device = device
        self.batch_size = batch_size

        self.model = create_surrogate_rf(
            surrogate_type=surrogate_type,
            seed=seed,
            n_estimators=resolved_n_trees,
        )

        self.uq_model: GPUProximityRegressionUQ | None = None
        self.shaker: EpistemicQuantifier | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.oob_res_: np.ndarray | None = None
        self.cv_res_: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> MultiUQEvaluator:
        """Fit the forest and the UQ engines and compute OOB and 5-fold CV residuals."""
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.float64).flatten()

        if X_arr.ndim != 2:
            raise ValueError(f"X_train must be a 2D array of shape (N, D), got {X_arr.ndim}D array.")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(f"X_train cannot have empty dimensions, got shape {X_arr.shape}.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Sample size mismatch: X_train has {X_arr.shape[0]} samples, "
                f"but y_train has {y_arr.shape[0]} samples."
            )

        N_train = X_arr.shape[0]
        self.model.fit(X_arr, y_arr)

        self.uq_model = GPUProximityRegressionUQ(
            model=self.model,
            X_train=X_arr,
            y_train=y_arr,
            device=self.device,
            topological_decay_lambda=self.topological_decay_lambda,
            density_scaling_alpha=self.density_scaling_alpha,
            use_density_scaling=True,
            weighting="unweighted_all",
            weighted=False,
        )
        self.uq_model.fit()

        if self.eval_mode == "unweighted_proximity_only":
            self.shaker = None
        else:
            self.shaker = EpistemicQuantifier(
                model=self.model,
                X_train=X_arr,
                y_train=y_arr,
            )

        # Points that were never out-of-bag get a zero residual.
        oob_pred = np.asarray(self.model.oob_prediction_, dtype=np.float64).ravel()
        clean_oob_pred = np.where(np.isfinite(oob_pred), oob_pred, y_arr)
        self.oob_res_ = (y_arr - clean_oob_pred).astype(np.float64)

        if N_train < 5:
            warnings.warn(
                f"Training set size N={N_train} is smaller than 5 CV folds. "
                "Falling back to OOB residuals for CV.",
                UserWarning,
                stacklevel=2,
            )
            self.cv_res_ = self.oob_res_.copy()
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_res = np.zeros(N_train, dtype=np.float64)
            for train_idx, val_idx in kf.split(X_arr):
                sub_model = clone(self.model)
                sub_model.set_params(oob_score=False)
                sub_model.fit(X_arr[train_idx], y_arr[train_idx])
                y_pred_val = sub_model.predict(X_arr[val_idx])
                cv_res[val_idx] = y_arr[val_idx] - np.asarray(y_pred_val, dtype=np.float64).ravel()
            self.cv_res_ = cv_res

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self._is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray) -> MultiUQResult:
        """Evaluate all approaches on ``X_test`` (shape ``(M, D)`` or ``(D,)``)."""
        if not self._is_fitted or self.X_train_ is None:
            raise RuntimeError("MultiUQEvaluator must be fitted before calling evaluate().")

        X_test_arr = np.asarray(X_test, dtype=np.float64)
        if X_test_arr.ndim == 1:
            X_test_2d = X_test_arr[np.newaxis, :]
        elif X_test_arr.ndim == 2:
            X_test_2d = X_test_arr
        else:
            raise ValueError(f"X_test must be a 1D or 2D array, got {X_test_arr.ndim}D array.")

        if X_test_2d.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: query X_test has {X_test_2d.shape[1]} features, "
                f"but training data had {self.X_train_.shape[1]} features."
            )

        M = X_test_2d.shape[0]
        N_train = len(self.X_train_)
        estimators = self.model.estimators_
        B = len(estimators)

        # Hutter: law of total variance over trees (between-tree + mean leaf impurity).
        tree_preds = np.column_stack([t.predict(X_test_2d) for t in estimators])
        y_hat = np.mean(tree_preds, axis=1).astype(np.float64)

        var_between = np.var(tree_preds, axis=1, ddof=1).astype(np.float64) if B > 1 else np.zeros_like(y_hat)

        tree_impurities = np.column_stack(
            [np.maximum(0.0, t.tree_.impurity[t.apply(X_test_2d)]) for t in estimators]
        )
        var_within = np.mean(tree_impurities, axis=1).astype(np.float64)

        var_between = np.maximum(0.0, var_between)
        var_within = np.maximum(0.0, var_within)
        var_total = np.maximum(0.0, var_between + var_within)

        sigma_hutter_total = np.sqrt(var_total).astype(np.float64)
        u_hutter_total = (self.kappa * sigma_hutter_total).astype(np.float64)
        u_hutter_between = (self.kappa * np.sqrt(var_between)).astype(np.float64)
        u_hutter_within = (self.kappa * np.sqrt(var_within)).astype(np.float64)

        # Shaker: epistemic std from the Huber mutual information, scaled by the aleatoric std.
        if self.eval_mode == "unweighted_proximity_only":
            sigma_shaker_epistemic = np.zeros(M, dtype=np.float64)
            u_shaker_epistemic = np.zeros(M, dtype=np.float64)
            shaker_intervals = {
                f"int_{side}_shaker_epistemic_{sfx}": y_hat.copy()
                for sfx in _GAUSSIAN_QUANTILES
                for side in ("lwr", "upr")
            }
            shaker_mi = np.zeros(M, dtype=np.float64)
            shaker_total_entropy = np.zeros(M, dtype=np.float64)
            u_shaker_total = np.zeros(M, dtype=np.float64)
        else:
            shaker_mi = np.asarray(
                self.shaker.shaker_get_epistemic_entropy(
                    X_test_2d,
                    method="huber",
                    backend=self.device,
                ),
                dtype=np.float64,
            )

            shaker_total_entropy = np.asarray(
                self.shaker._shaker_calc_total_entropy(
                    X_test_2d,
                    method="huber",
                    backend=self.device,
                ),
                dtype=np.float64,
            )

            aleatoric_var = np.asarray(
                self.shaker.base_get_aleatoric_variance(X_test_2d),
                dtype=np.float64,
            )
            sigma_aleatoric = np.sqrt(np.maximum(aleatoric_var, 1e-6))

            safe_mi_exp = np.clip(2.0 * shaker_mi, 0.0, 50.0)
            sigma_epistemic = sigma_aleatoric * np.sqrt(np.maximum(2.0**safe_mi_exp - 1.0, 0.0))
            sigma_shaker_epistemic = sigma_epistemic.astype(np.float64)
            u_shaker_epistemic = (self.kappa * sigma_shaker_epistemic).astype(np.float64)
            shaker_intervals = _gaussian_intervals("shaker_epistemic", y_hat, sigma_shaker_epistemic)

            safe_tot_exp = np.clip(2.0 * shaker_total_entropy, -50.0, 50.0)
            entropy_power_var = (2.0**safe_tot_exp) / (2.0 * np.pi * np.e)
            u_shaker_total = (self.kappa * np.sqrt(np.maximum(entropy_power_var, 0.0))).astype(np.float64)

        # Proximity methods: residual quantiles weighted by tree-path proximity.
        k_eff = min(self.k, N_train)

        u_prox: dict[str, np.ndarray] = {app: np.zeros(M, dtype=np.float64) for app in PROXIMITY_APPROACHES}
        int_lwr_prox: dict[str, dict[str, np.ndarray]] = {
            app: {sfx: np.zeros(M, dtype=np.float64) for sfx in ("50", "80", "90", "95")}
            for app in PROXIMITY_APPROACHES
        }
        int_upr_prox: dict[str, dict[str, np.ndarray]] = {
            app: {sfx: np.zeros(M, dtype=np.float64) for sfx in ("50", "80", "90", "95")}
            for app in PROXIMITY_APPROACHES
        }

        q_lower = np.zeros(M, dtype=np.float64)
        delta_floor = np.zeros(M, dtype=np.float64)
        local_mae = np.zeros(M, dtype=np.float64)
        u_rf_fire_lower = np.zeros(M, dtype=np.float64)

        oob_res = np.asarray(self.oob_res_, dtype=np.float64)
        cv_res = np.asarray(self.cv_res_, dtype=np.float64)
        valid_oob_mask = np.asarray(self.uq_model.valid_oob_mask)
        n_baseline = float(getattr(self.uq_model, "N_baseline", 1.0))
        chunk_size = 512 if isinstance(self.batch_size, str) else int(self.batch_size)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            b_len = end - start
            X_chunk = X_test_2d[start:end, :]
            y_hat_chunk = y_hat[start:end]
            leaf_batch = self.model.apply(X_chunk)

            prox_topo_unweighted = np.zeros((b_len, N_train), dtype=np.float32)
            density_chunk = np.zeros(b_len, dtype=np.float32)
            prox_fire = np.zeros((b_len, N_train), dtype=np.float32)

            for t in range(B):
                # ``.get()`` copies CuPy arrays back to the host when the GPU backend is used.
                id_to_dense = self.uq_model.tree_leaf_id_to_dense[t]
                if hasattr(id_to_dense, "get"):
                    id_to_dense = id_to_dense.get()
                dense_test = id_to_dense[leaf_batch[:, t]]
                dense_train_inbag = id_to_dense[self.uq_model.in_bag_leaves[:, t]]
                dense_train_all = id_to_dense[self.uq_model.leaf_matrix_train[:, t]]

                tree_dists = self.uq_model.tree_leaf_distances[t]
                if hasattr(tree_dists, "get"):
                    tree_dists = tree_dists.get()

                d_t_unweighted = tree_dists[dense_test[:, None], dense_train_all[None, :]]
                decay_t_unweighted = np.exp(-self.topological_decay_lambda * d_t_unweighted)
                prox_topo_unweighted += decay_t_unweighted

                d_t_inbag = tree_dists[dense_test[:, None], dense_train_inbag[None, :]]
                decay_t_inbag = np.exp(-self.topological_decay_lambda * d_t_inbag)
                in_bag_c = self.uq_model.in_bag_counts[:, t]
                density_chunk += np.sum(decay_t_inbag * in_bag_c[None, :], axis=1)

                train_w = self.uq_model.train_weights[:, t]
                matches_t = leaf_batch[:, t, None] == self.uq_model.in_bag_leaves[None, :, t]
                prox_fire += matches_t * train_w[None, :]

            prox_topo_unweighted /= B
            prox_fire /= B

            avg_test_leaf_sizes = density_chunk / B
            avg_test_leaf_sizes = np.maximum(avg_test_leaf_sizes, 1e-5)
            gamma = np.maximum(1.0, (n_baseline / avg_test_leaf_sizes) ** self.density_scaling_alpha)

            for pool in ("oob", "cv"):
                pool_residuals = oob_res if pool == "oob" else cv_res

                if pool == "oob":
                    prox_pool = prox_topo_unweighted.copy()
                    prox_pool[:, ~valid_oob_mask] = 0.0
                else:
                    prox_pool = prox_topo_unweighted

                # Method A: empirical quantiles of the k most proximal residuals.
                if k_eff < N_train:
                    partition_idx = np.flip(np.argsort(prox_pool, axis=1), axis=1)[:, :k_eff]
                    k_residuals = pool_residuals[partition_idx]
                else:
                    k_residuals = np.broadcast_to(pool_residuals[None, :], (b_len, N_train))

                app_a = f"prox_a_{pool}"
                app_ac = f"prox_ac_{pool}"
                app_b = f"prox_b_{pool}"
                app_bc = f"prox_bc_{pool}"

                for sfx, a_lwr, a_upr in _RESIDUAL_QUANTILES:
                    q_lwr_a = np.quantile(k_residuals, a_lwr, axis=1)
                    q_upr_a = np.quantile(k_residuals, a_upr, axis=1)

                    int_lwr_prox[app_a][sfx][start:end] = y_hat_chunk + q_lwr_a
                    int_upr_prox[app_a][sfx][start:end] = y_hat_chunk + q_upr_a

                    int_lwr_prox[app_ac][sfx][start:end] = y_hat_chunk + gamma * q_lwr_a
                    int_upr_prox[app_ac][sfx][start:end] = y_hat_chunk + gamma * q_upr_a

                    # Intervals are uncapped; the exploration term is capped at zero.
                    if sfx == "95":
                        u_prox[app_a][start:end] = np.maximum(0.0, -q_lwr_a)
                        u_prox[app_ac][start:end] = gamma * np.maximum(0.0, -q_lwr_a)

                    # Method B: proximity-weighted quantiles over all residuals.
                    q_lwr_b = self.uq_model._compute_weighted_quantile(pool_residuals, prox_pool, a_lwr)
                    q_upr_b = self.uq_model._compute_weighted_quantile(pool_residuals, prox_pool, a_upr)
                    if hasattr(q_lwr_b, "get"):
                        q_lwr_b = q_lwr_b.get()
                    if hasattr(q_upr_b, "get"):
                        q_upr_b = q_upr_b.get()
                    q_lwr_b = np.asarray(q_lwr_b, dtype=np.float64)
                    q_upr_b = np.asarray(q_upr_b, dtype=np.float64)

                    int_lwr_prox[app_b][sfx][start:end] = y_hat_chunk + q_lwr_b
                    int_upr_prox[app_b][sfx][start:end] = y_hat_chunk + q_upr_b

                    int_lwr_prox[app_bc][sfx][start:end] = y_hat_chunk + gamma * q_lwr_b
                    int_upr_prox[app_bc][sfx][start:end] = y_hat_chunk + gamma * q_upr_b

                    if sfx == "95":
                        u_prox[app_b][start:end] = np.maximum(0.0, -q_lwr_b)
                        u_prox[app_bc][start:end] = gamma * np.maximum(0.0, -q_lwr_b)

                if pool == "oob":
                    # Diagnostics from the 95 % OOB method-A interval.
                    q_lwr_a_95 = np.quantile(k_residuals, 0.025, axis=1)
                    q_upr_a_95 = np.quantile(k_residuals, 0.975, axis=1)
                    in_int_topo_u = (k_residuals >= q_lwr_a_95[:, None]) & (
                        k_residuals <= q_upr_a_95[:, None]
                    )
                    cnt_topo_u = np.sum(in_int_topo_u, axis=1)
                    sum_abs_topo_u = np.sum(np.abs(k_residuals) * in_int_topo_u, axis=1)
                    b_local_mae_u = np.where(
                        cnt_topo_u > 0,
                        sum_abs_topo_u / np.maximum(cnt_topo_u, 1),
                        float(self.uq_model.oob_mae) if hasattr(self.uq_model, "oob_mae") else 1.0,
                    )
                    b_delta_floor_u = self.epsilon * self.kappa * b_local_mae_u

                    q_lower[start:end] = q_lwr_a_95
                    delta_floor[start:end] = b_delta_floor_u
                    local_mae[start:end] = b_local_mae_u

            # RF-FIRE: k nearest residuals under shared-leaf (non-topological) proximity.
            prox_fire_pool = prox_fire.copy()
            prox_fire_pool[:, ~valid_oob_mask] = 0.0

            if k_eff < N_train:
                partition_idx_fire = np.flip(np.argsort(prox_fire_pool, axis=1), axis=1)[:, :k_eff]
                k_residuals_fire = oob_res[partition_idx_fire]
            else:
                k_residuals_fire = np.broadcast_to(oob_res[None, :], (b_len, N_train))
            q_lwr_fire = np.quantile(k_residuals_fire, 0.025, axis=1)
            u_rf_fire_lower[start:end] = np.maximum(0.0, -q_lwr_fire)

        result_kwargs: dict[str, Any] = {
            "u_hutter_total": u_hutter_total,
            "sigma_hutter_total": sigma_hutter_total,
            **_gaussian_intervals("hutter_total", y_hat, sigma_hutter_total),
            "u_shaker_epistemic": u_shaker_epistemic,
            "sigma_shaker_epistemic": sigma_shaker_epistemic,
            **shaker_intervals,
            "var_between": var_between,
            "var_within": var_within,
            "var_total": var_total,
            "delta_floor": delta_floor,
            "local_mae": local_mae,
            "q_lower": q_lower,
            "u_hutter_between": u_hutter_between,
            "u_hutter_within": u_hutter_within,
            "u_shaker_total": u_shaker_total,
            "shaker_mi": shaker_mi,
            "shaker_total_entropy": shaker_total_entropy,
            "u_rf_fire_lower": u_rf_fire_lower,
        }
        for app in PROXIMITY_APPROACHES:
            result_kwargs[f"u_{app}"] = u_prox[app]
            result_kwargs[f"sigma_{app}"] = (u_prox[app] / 1.96).astype(np.float64)
            for sfx in ("50", "80", "90", "95"):
                result_kwargs[f"int_lwr_{app}_{sfx}"] = int_lwr_prox[app][sfx]
                result_kwargs[f"int_upr_{app}_{sfx}"] = int_upr_prox[app][sfx]

        return MultiUQResult(y_hat, **result_kwargs)
