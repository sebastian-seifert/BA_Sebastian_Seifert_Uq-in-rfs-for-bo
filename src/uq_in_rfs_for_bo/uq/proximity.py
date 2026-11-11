"""Proximity-based local uncertainty for random forests (NumPy or CuPy backend).

Prediction intervals are empirical quantiles of out-of-bag (or cross-validation) residuals of the
training points, weighted by their forest proximity to the query point. Proximities are either leaf
co-occurrence or a topological ``exp(-lambda * d)`` decay over tree-path distances ``d``, optionally
combined with a leaf-density scaling of the interval width.
"""

import sys
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble._forest import _generate_sample_indices, _generate_unsampled_indices
from sklearn.model_selection import KFold

# Cluster jobs may start with a non-UTF-8 locale; avoid UnicodeEncodeErrors in log output.
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
if sys.stderr is not None:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

try:
    import cupy as cp
    import cupyx

    HAS_CUPY = True
    try:
        HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        HAS_GPU = False
except ImportError:
    cp = None
    cupyx = None
    HAS_CUPY = False
    HAS_GPU = False

_WEIGHTING_OPTIONS = {"leaf_normalized", "weighted", "unweighted_all", "unweighted", "unweighted_inbag"}


class GPUProximityRegressionUQ:
    """Proximity-weighted residual quantiles of a random forest.

    Parameters
    ----------
    model : RandomForestRegressor
        Forest to wrap; fitted here (with ``oob_score=True``) if it is not fitted yet.
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    device : {"auto", "cpu", "gpu", "cuda"}, default="auto"
        "auto" uses CuPy when a GPU is available.
    batch_size : int or "auto", default="auto"
        Number of query points processed at once; "auto" derives it from free memory.
    use_density_scaling : bool, default=False
        Multiply the interval width by ``gamma(x) = max(1, (N_baseline / n(x))^alpha)``, where
        ``n(x)`` is the (walked) leaf population at ``x``.
    density_scaling_alpha : float, default=1.0
        Exponent ``alpha`` of the density scaling.
    topological_decay_lambda : float or None, default=None
        Decay rate of the tree-path proximity; None or 0 selects leaf co-occurrence.
    weighting : str, default="unweighted_all"
        "unweighted_all" (all training points), "unweighted_inbag" (in-bag points only) or
        "leaf_normalized" (in-bag counts divided by leaf size). "unweighted" and "weighted" are
        aliases of the first and last.
    use_leaf_weights : bool or None, default=None
        True forces "leaf_normalized"; False turns a leaf-normalised weighting into "unweighted_all".
    weighted : bool, default=False
        True forces "leaf_normalized".
    residual_mode : {"oob", "cv"}, default="oob"
        Residuals used for the quantiles: out-of-bag or K-fold cross-validation.
    cv_folds : int, default=5
        Number of folds for ``residual_mode="cv"``.
    fallback_on_low_support : bool, default=False
        Default for the per-call option of the same name in :meth:`predict_with_intervals`.
    """

    def __init__(
        self,
        model,
        X_train,
        y_train,
        device="auto",
        batch_size="auto",
        use_density_scaling=False,
        density_scaling_alpha=1.0,
        topological_decay_lambda=None,
        weighting: str = "unweighted_all",
        use_leaf_weights: bool | None = None,
        weighted: bool = False,
        residual_mode: str = "oob",
        cv_folds: int = 5,
        fallback_on_low_support: bool = False,
    ):
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.n_train = len(self.X_train)
        self.batch_size_param = batch_size
        self.use_density_scaling = use_density_scaling
        self.density_scaling_alpha = density_scaling_alpha
        self.topological_decay_lambda = topological_decay_lambda

        valid_residual_modes = {"oob", "cv"}
        if residual_mode not in valid_residual_modes:
            raise ValueError(
                f"Unknown residual_mode '{residual_mode}'. Expected one of {sorted(valid_residual_modes)}"
            )
        self.residual_mode = residual_mode
        self.cv_folds = int(cv_folds)
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be an integer >= 2, got {self.cv_folds}.")

        if weighting not in _WEIGHTING_OPTIONS:
            raise ValueError(
                f"Unknown weighting scheme '{weighting}'. Expected one of {sorted(_WEIGHTING_OPTIONS)}"
            )

        if weighted or use_leaf_weights is True:
            weighting = "leaf_normalized"
        elif use_leaf_weights is False and weighting in ("leaf_normalized", "weighted"):
            weighting = "unweighted_all"

        if weighting in ("unweighted", "unweighted_all"):
            weighting = "unweighted_all"
        elif weighting in ("weighted", "leaf_normalized"):
            weighting = "leaf_normalized"

        self.weighting = weighting
        self.use_leaf_weights = use_leaf_weights
        self.weighted = self.weighting == "leaf_normalized"
        self.fallback_on_low_support = bool(fallback_on_low_support)
        self.is_fitted = False

        self._init_backend(device)

        if hasattr(self.model, "estimators_"):
            self.fit()

    def _init_backend(self, device):
        """Select NumPy or CuPy as array backend ``self.xp``."""
        device = device.lower()
        self.using_gpu = False

        if device in ("gpu", "cuda"):
            if HAS_GPU:
                self.using_gpu = True
            elif HAS_CUPY:
                warnings.warn(
                    "CuPy is installed but no GPUs were detected. Falling back to CPU.",
                    UserWarning,
                    stacklevel=1,
                )
            else:
                warnings.warn(
                    "GPU backend requested but CuPy is not installed. Falling back to CPU.",
                    UserWarning,
                    stacklevel=1,
                )
        elif device == "auto":
            if HAS_GPU:
                self.using_gpu = True

        if self.using_gpu:
            self.xp = cp
            # Older CuPy versions lack nanquantile.
            try:
                _ = cp.nanquantile(cp.array([1.0, cp.nan]), 0.5)
                self.nanquantile_supported = True
            except (AttributeError, NotImplementedError):
                self.nanquantile_supported = False
        else:
            self.xp = np
            self.nanquantile_supported = True

    def _get_dynamic_batch_size(self, n_test):
        """Batch size bounded by free GPU memory, or by host memory and cache size on the CPU."""
        if not self.using_gpu:
            available_bytes = 4 * 1024 * 1024 * 1024
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemAvailable" in line:
                            parts = line.split()
                            available_bytes = int(parts[1]) * 1024
                            break
            except Exception:
                pass

            # Budget min(50 MB, 1% of available RAM) for the (batch, n_train, n_estimators) work.
            target_bytes = min(50_000_000, int(available_bytes * 0.01))

            total_elements = self.n_train * self.n_estimators
            if total_elements <= 0:
                return 128
            opt_batch = target_bytes // total_elements
            return max(1, min(n_test, min(128, opt_batch)))

        try:
            free_mem, total_mem = cp.cuda.Device().mem_info
            # ~12 bytes per (test, train) pair: float32 proximity, bool mask and temporaries.
            bytes_per_sample = 12 * self.n_train
            if bytes_per_sample <= 0:
                return 256
            usable_vram = 0.8 * free_mem
            opt_batch = int(usable_vram // bytes_per_sample)
            # Cap at 512 so that several processes can share one GPU.
            return max(1, min(n_test, min(512, opt_batch)))
        except Exception:
            return 256

    def _resolve_batch_size(self, n_test):
        if self.batch_size_param == "auto":
            return self._get_dynamic_batch_size(n_test)
        return int(self.batch_size_param)

    def fit(self):
        """Fit the forest if needed and precompute residuals, leaf weights, distances and densities."""
        if not hasattr(self.model, "estimators_"):
            self.model.set_params(oob_score=True)
            self.model.fit(self.X_train, self.y_train)
        elif not getattr(self.model, "oob_score", False):
            warnings.warn(
                "The provided RandomForestRegressor was fitted without oob_score=True. "
                "Refitting to extract OOB stats.",
                UserWarning,
                stacklevel=1,
            )
            self.model.set_params(oob_score=True)
            self.model.fit(self.X_train, self.y_train)

        self.estimators = self.model.estimators_
        self.n_estimators = len(self.estimators)
        self.oob_prediction_ = self.model.oob_prediction_

        # Reconstruct the bootstrap sample of every tree.
        if self.using_gpu:
            self.oob_indices = cupyx.zeros_pinned((self.n_train, self.n_estimators), dtype=np.int32)
            self.in_bag_counts = cupyx.zeros_pinned((self.n_train, self.n_estimators), dtype=np.int32)
        else:
            self.oob_indices = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)
            self.in_bag_counts = np.zeros((self.n_train, self.n_estimators), dtype=np.int32)

        for t, tree in enumerate(self.estimators):
            if hasattr(self.model, "estimators_samples_") and t < len(self.model.estimators_samples_):
                ib_idx = self.model.estimators_samples_[t]
                in_bag_set = set(ib_idx)
                oob_idx = np.array([i for i in range(self.n_train) if i not in in_bag_set], dtype=np.int32)
            else:
                max_samples = getattr(self.model, "max_samples", None)
                if max_samples is None:
                    n_samples_bootstrap = self.n_train
                elif isinstance(max_samples, (int, np.integer)):
                    n_samples_bootstrap = int(max_samples)
                elif isinstance(max_samples, float):
                    n_samples_bootstrap = int(round(max_samples * self.n_train))
                else:
                    n_samples_bootstrap = self.n_train
                oob_idx = _generate_unsampled_indices(tree.random_state, self.n_train, n_samples_bootstrap)
                ib_idx = _generate_sample_indices(tree.random_state, self.n_train, n_samples_bootstrap)

            self.oob_indices[oob_idx, t] = 1
            idx, counts = np.unique(ib_idx, return_counts=True)
            self.in_bag_counts[idx, t] = counts

        if self.residual_mode == "cv":
            raw_residuals = self._compute_cv_residuals()
            if self.n_train >= self.cv_folds:
                self.valid_oob_mask = np.ones(self.n_train, dtype=bool)
        else:
            raw_residuals = self._compute_oob_residuals()

        if self.using_gpu:
            self.oob_residuals = cupyx.empty_pinned(raw_residuals.shape, dtype=np.float32)
            self.oob_residuals[...] = raw_residuals
        else:
            self.oob_residuals = raw_residuals.astype(np.float32)

        # in_bag_leaves holds the leaf id for in-bag samples and 0 (the root) for OOB samples.
        if self.using_gpu:
            self.in_bag_indices = cupyx.empty_pinned(self.oob_indices.shape, dtype=np.int32)
            np.subtract(1, self.oob_indices, out=self.in_bag_indices)

            leaf_matrix_train_cpu = self.model.apply(self.X_train)
            self.leaf_matrix_train = cupyx.empty_pinned(
                leaf_matrix_train_cpu.shape, dtype=leaf_matrix_train_cpu.dtype
            )
            self.leaf_matrix_train[...] = leaf_matrix_train_cpu

            self.in_bag_leaves = cupyx.empty_pinned(
                self.leaf_matrix_train.shape, dtype=self.leaf_matrix_train.dtype
            )
            np.multiply(self.in_bag_indices, self.leaf_matrix_train, out=self.in_bag_leaves)
        else:
            self.in_bag_indices = 1 - self.oob_indices
            self.leaf_matrix_train = self.model.apply(self.X_train)
            self.in_bag_leaves = self.in_bag_indices * self.leaf_matrix_train

        # Leaf-normalised training weights: in-bag count / in-bag leaf size.
        if self.using_gpu:
            self.train_weights = cupyx.zeros_pinned((self.n_train, self.n_estimators), dtype=np.float32)
        else:
            self.train_weights = np.zeros((self.n_train, self.n_estimators), dtype=np.float32)

        self.leaf_sizes = []
        for t, tree in enumerate(self.estimators):
            node_count = tree.tree_.node_count
            leaf_sums = np.bincount(
                self.in_bag_leaves[:, t], weights=self.in_bag_counts[:, t], minlength=node_count
            )
            # Nodes without in-bag samples (including the root slot used for OOB rows) get size 1.
            leaf_sums[leaf_sums == 0.0] = 1.0
            self.leaf_sizes.append(self.xp.asarray(leaf_sums))

            train_leaves = self.leaf_matrix_train[:, t]
            self.train_weights[:, t] = (self.in_bag_indices[:, t] * self.in_bag_counts[:, t]) / leaf_sums[
                train_leaves
            ]

        self.oob_residuals_xp = self.xp.asarray(self.oob_residuals)
        self.in_bag_leaves_xp = self.xp.asarray(self.in_bag_leaves)
        self.train_weights_xp = self.xp.asarray(self.train_weights)
        self.in_bag_counts_xp = self.xp.asarray(self.in_bag_counts)
        self.leaf_matrix_train_xp = self.xp.asarray(self.leaf_matrix_train)
        self.in_bag_indices_xp = self.xp.asarray(self.in_bag_indices, dtype=self.xp.float32)

        self._precompute_tree_distances()
        self._compute_baseline_density()
        self.is_fitted = True

    def _compute_oob_residuals(self) -> np.ndarray:
        """OOB residuals ``y - y_oob``; sets ``valid_oob_mask`` and zeroes invalid entries."""
        oob_counts = np.sum(self.oob_indices, axis=1)
        self.valid_oob_mask = (oob_counts > 0) & (
            np.isfinite(self.oob_prediction_) if self.oob_prediction_ is not None else False
        )

        # Rows without a valid OOB prediction get residual 0 so they cannot inject NaNs.
        if self.oob_prediction_ is not None:
            clean_oob_pred = np.where(self.valid_oob_mask, self.oob_prediction_, self.y_train)
            raw_residuals = self.y_train - clean_oob_pred
        else:
            raw_residuals = np.zeros_like(self.y_train)
        return raw_residuals

    def _compute_cv_residuals(self) -> np.ndarray:
        """Out-of-fold K-fold residuals; falls back to OOB residuals when ``n_train < cv_folds``."""
        if self.n_train < self.cv_folds:
            warnings.warn(
                f"Training set size N={self.n_train} is smaller than cv_folds={self.cv_folds}. "
                "Falling back to OOB residuals.",
                UserWarning,
                stacklevel=2,
            )
            return self._compute_oob_residuals()

        seed = getattr(self.model, "random_state", None)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=seed)

        y_train_flat = np.asarray(self.y_train, dtype=np.float32).ravel()
        raw_residuals = np.zeros(self.n_train, dtype=np.float32)
        for train_idx, val_idx in kf.split(self.X_train):
            sub_model = clone(self.model)
            if hasattr(sub_model, "oob_score"):
                sub_model.set_params(oob_score=False)
            sub_model.fit(self.X_train[train_idx], self.y_train[train_idx])
            y_pred_val = sub_model.predict(self.X_train[val_idx])
            if isinstance(y_pred_val, tuple):
                y_pred_val = y_pred_val[0]
            y_pred_val = np.asarray(y_pred_val, dtype=np.float32).flatten()
            raw_residuals[val_idx] = y_train_flat[val_idx] - y_pred_val

        self.valid_oob_mask = np.ones(self.n_train, dtype=bool)
        return raw_residuals

    def _precompute_tree_distances(self):
        """Leaf-to-leaf path distance matrix of every tree, indexed by dense leaf number."""
        if hasattr(self, "tree_leaf_distances") and self.tree_leaf_distances is not None:
            return

        self.tree_leaf_distances = []
        self.tree_leaf_id_to_dense = []

        for t in range(self.n_estimators):
            tree = self.estimators[t].tree_
            leaf_nodes = np.where(tree.children_left == -1)[0].astype(np.int32)

            leaf_nodes_xp = self.xp.asarray(leaf_nodes)
            self.tree_leaf_distances.append(
                self.compute_tree_topological_distances(leaf_nodes_xp, leaf_nodes_xp, t)
            )

            # Internal nodes map to -1, which indexes the last leaf; such entries are always
            # multiplied by a zero in-bag count or weight.
            id_to_dense = np.full(tree.node_count, -1, dtype=np.int32)
            for dense_idx, leaf_id in enumerate(leaf_nodes):
                id_to_dense[leaf_id] = dense_idx
            self.tree_leaf_id_to_dense.append(self.xp.asarray(id_to_dense))

    def _compute_baseline_density(self) -> None:
        """Median training density ``N_baseline`` used as reference by the density scaling."""
        if self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0:
            # Subsample at most 250 training points to bound memory.
            if self.n_train > 250:
                np.random.seed(42)
                sub_indices = np.random.choice(self.n_train, 250, replace=False)
            else:
                sub_indices = np.arange(self.n_train)

            sub_n = len(sub_indices)
            sub_leaf_matrix_xp = self.xp.asarray(self.leaf_matrix_train[sub_indices, :])

            train_walked_densities = self.xp.zeros(sub_n, dtype=self.xp.float32)
            for t in range(self.n_estimators):
                id_to_dense = self.tree_leaf_id_to_dense[t]
                dense_test = id_to_dense[sub_leaf_matrix_xp[:, t]]
                dense_train = id_to_dense[self.in_bag_leaves_xp[:, t]]
                d_t = self.tree_leaf_distances[t][dense_test[:, None], dense_train[None, :]]
                decay_t = self.xp.exp(-self.topological_decay_lambda * d_t)
                train_walked_densities += self.xp.sum(decay_t * self.in_bag_counts_xp[None, :, t], axis=1)

            train_walked_densities /= self.n_estimators
            self.N_baseline = float(self.xp.median(train_walked_densities))
        else:
            train_leaf_sizes = self.xp.zeros((self.n_train, self.n_estimators), dtype=self.xp.float32)
            for t in range(self.n_estimators):
                train_leaf_sizes[:, t] = self.leaf_sizes[t][self.xp.asarray(self.leaf_matrix_train[:, t])]
            self.train_avg_leaf_sizes = self.xp.mean(train_leaf_sizes, axis=1)
            self.N_baseline = float(self.xp.median(self.train_avg_leaf_sizes))

    def _resolve_runtime_weighting(
        self,
        weighting: str | None = None,
        use_leaf_weights: bool | None = None,
        weighted: bool | None = None,
    ) -> str:
        """Canonical weighting for a call, applying the same rules as the constructor."""
        if weighting is None and use_leaf_weights is None and weighted is None:
            return self.weighting
        w = weighting if weighting is not None else self.weighting
        if w not in _WEIGHTING_OPTIONS:
            raise ValueError(f"Unknown weighting scheme '{w}'. Expected one of {sorted(_WEIGHTING_OPTIONS)}")
        if weighted is True or use_leaf_weights is True:
            return "leaf_normalized"
        elif (weighted is False or use_leaf_weights is False) and w in ("leaf_normalized", "weighted"):
            return "unweighted_all"
        if w in ("unweighted", "unweighted_all"):
            return "unweighted_all"
        elif w in ("weighted", "leaf_normalized"):
            return "leaf_normalized"
        return w

    def _parse_n_neighbors(self, n_neighbors, warn_on_float: bool):
        """Validate ``n_neighbors``: "auto", "all" or an integer in [1, n_train] (floats are rounded)."""
        if n_neighbors == "auto" or n_neighbors == "all":
            return n_neighbors
        try:
            if isinstance(n_neighbors, float) or (isinstance(n_neighbors, str) and "." in n_neighbors):
                n_neighbors_val = int(round(float(n_neighbors)))
                if warn_on_float:
                    warnings.warn(
                        f"n_neighbors value {n_neighbors} is a float/float-string. "
                        f"Rounding to integer: {n_neighbors_val}.",
                        UserWarning,
                        stacklevel=1,
                    )
            else:
                n_neighbors_val = int(n_neighbors)
            if n_neighbors_val <= 0 or n_neighbors_val > self.n_train:
                raise ValueError(f"n_neighbors must be between 1 and {self.n_train}.")
            return n_neighbors_val
        except (ValueError, TypeError):
            raise ValueError("n_neighbors must be a positive integer, 'auto', or 'all'.") from None

    def _test_leaves(self, X_test):
        """Leaf ids of ``X_test`` (n_test, n_estimators) on the active backend."""
        leaf_matrix_test = self.model.apply(X_test)
        if self.using_gpu:
            # Stage through pinned host memory for a fast host-to-device copy.
            leaf_matrix_test_pinned = cupyx.empty_pinned(leaf_matrix_test.shape, dtype=leaf_matrix_test.dtype)
            leaf_matrix_test_pinned[...] = leaf_matrix_test
            return cp.asarray(leaf_matrix_test_pinned)
        return leaf_matrix_test

    def _proximity_batch(self, leaf_batch, weighting, topological=True, density_batch=None):
        """Proximity of each query leaf row to every training point, shape (batch, n_train).

        Uses the topological decay kernel when ``topological`` and ``topological_decay_lambda > 0``,
        otherwise leaf co-occurrence. Columns of training points without a valid residual are zeroed.
        If ``density_batch`` is given, the walked in-bag density is accumulated into it in place.
        """
        xp = self.xp
        prox_batch = xp.zeros((len(leaf_batch), self.n_train), dtype=xp.float32)
        use_decay = (
            topological and self.topological_decay_lambda is not None and self.topological_decay_lambda > 0.0
        )

        for t in range(self.n_estimators):
            if use_decay:
                id_to_dense = self.tree_leaf_id_to_dense[t]
                dense_test = id_to_dense[leaf_batch[:, t]]
                if weighting == "unweighted_all":
                    dense_train = id_to_dense[self.leaf_matrix_train_xp[:, t]]
                else:
                    dense_train = id_to_dense[self.in_bag_leaves_xp[:, t]]
                d_t = self.tree_leaf_distances[t][dense_test[:, None], dense_train[None, :]]
                decay_t = xp.exp(-self.topological_decay_lambda * d_t)
                if weighting == "unweighted_all":
                    prox_batch += decay_t
                elif weighting == "unweighted_inbag":
                    prox_batch += decay_t * self.in_bag_indices_xp[None, :, t]
                else:
                    prox_batch += decay_t * self.train_weights_xp[None, :, t]

                if density_batch is not None:
                    density_batch += xp.sum(decay_t * self.in_bag_counts_xp[None, :, t], axis=1)
            elif weighting == "unweighted_all":
                matches_t = leaf_batch[:, t, None] == self.leaf_matrix_train_xp[None, :, t]
                prox_batch += matches_t.astype(xp.float32)
            elif weighting == "unweighted_inbag":
                matches_t = leaf_batch[:, t, None] == self.in_bag_leaves_xp[None, :, t]
                prox_batch += matches_t.astype(xp.float32) * self.in_bag_indices_xp[None, :, t]
            else:
                matches_t = leaf_batch[:, t, None] == self.in_bag_leaves_xp[None, :, t]
                prox_batch += matches_t * self.train_weights_xp[None, :, t]

        prox_batch /= self.n_estimators
        if hasattr(self, "valid_oob_mask") and self.valid_oob_mask is not None:
            prox_batch[:, ~self.valid_oob_mask] = 0.0
        return prox_batch

    def _top_k_residuals(self, prox_batch, n_neighbors):
        """Residuals of the ``n_neighbors`` most proximate training points ("all": every point)."""
        k = self.n_train if n_neighbors == "all" else int(n_neighbors)
        if k < self.n_train:
            # argsort (not argpartition) keeps the tie-breaking of the reference implementation.
            partition_idx = self.xp.flip(self.xp.argsort(prox_batch, axis=1), axis=1)[:, :k]
            return self.oob_residuals_xp[partition_idx]
        return self.xp.broadcast_to(self.oob_residuals_xp[None, :], (len(prox_batch), self.n_train))

    def _interval_batch(
        self, leaf_batch, prox_batch, weighting, n_neighbors, alpha_lwr, alpha_upr, return_mae, fallback
    ):
        """Residual quantiles (and optionally the in-interval MAE) for one batch of query points.

        - topological proximity with ``n_neighbors`` "auto"/"all": proximity-weighted quantiles;
        - ``n_neighbors="auto"`` otherwise: quantiles over all points with non-zero proximity;
        - integer or "all": unweighted quantiles over the ``k`` most proximate points.

        With ``fallback``, rows whose effective neighbourhood size ``(sum w)^2 / sum w^2`` is below 3
        (or total weight below 1e-5) use the co-occurrence top-k interval instead.
        """
        xp = self.xp
        batch_mae = None
        if (
            self.topological_decay_lambda is not None
            and self.topological_decay_lambda > 0.0
            and n_neighbors
            in (
                "auto",
                "all",
            )
        ):
            batch_lwr = self._compute_weighted_quantile(self.oob_residuals_xp, prox_batch, alpha_lwr)
            batch_upr = self._compute_weighted_quantile(self.oob_residuals_xp, prox_batch, alpha_upr)
            if return_mae:
                resids = self.oob_residuals_xp[None, :]
                in_int = (resids >= batch_lwr[:, None]) & (resids <= batch_upr[:, None])
                w_in = prox_batch * in_int
                w_sum = xp.sum(w_in, axis=1)
                mae_val = xp.sum(xp.abs(resids) * w_in, axis=1) / xp.maximum(w_sum, 1e-10)
                batch_mae = xp.where(w_sum > 0, mae_val, float(self.oob_mae))
        elif n_neighbors == "auto":
            masked_residuals = xp.where(prox_batch >= 1e-10, self.oob_residuals_xp[None, :], xp.nan)
            if self.using_gpu and not self.nanquantile_supported:
                tiled_cpu = cp.asnumpy(masked_residuals)
                batch_lwr = cp.asarray(np.nanquantile(tiled_cpu, alpha_lwr, axis=1))
                batch_upr = cp.asarray(np.nanquantile(tiled_cpu, alpha_upr, axis=1))
            else:
                batch_lwr = xp.nanquantile(masked_residuals, alpha_lwr, axis=1)
                batch_upr = xp.nanquantile(masked_residuals, alpha_upr, axis=1)
            if return_mae:
                in_int = (masked_residuals >= batch_lwr[:, None]) & (masked_residuals <= batch_upr[:, None])
                valid_cnt = xp.sum(in_int & (~xp.isnan(masked_residuals)), axis=1)
                abs_res = xp.where(~xp.isnan(masked_residuals), xp.abs(masked_residuals), 0.0)
                sum_abs = xp.sum(abs_res * in_int, axis=1)
                mae_val = sum_abs / xp.maximum(valid_cnt, 1)
                batch_mae = xp.where(valid_cnt > 0, mae_val, float(self.oob_mae))
        else:
            k_residuals = self._top_k_residuals(prox_batch, n_neighbors)
            batch_lwr = xp.quantile(k_residuals, alpha_lwr, axis=1)
            batch_upr = xp.quantile(k_residuals, alpha_upr, axis=1)
            if return_mae:
                in_int = (k_residuals >= batch_lwr[:, None]) & (k_residuals <= batch_upr[:, None])
                cnt = xp.sum(in_int, axis=1)
                sum_abs = xp.sum(xp.abs(k_residuals) * in_int, axis=1)
                mae_val = sum_abs / xp.maximum(cnt, 1)
                batch_mae = xp.where(cnt > 0, mae_val, float(self.oob_mae))

        if fallback:
            w_sum = xp.sum(prox_batch, axis=1)
            w_sq_sum = xp.sum(prox_batch**2, axis=1)
            k_eff = xp.where(w_sq_sum > 0, (w_sum**2) / w_sq_sum, 0.0)
            low_support = (w_sum < 1e-5) | (k_eff < 3.0)
            if xp.any(low_support):
                low_idx = xp.where(low_support)[0]
                disc = self._compute_discrete_intervals_fallback(
                    leaf_batch[low_idx], weighting, n_neighbors, alpha_lwr, alpha_upr, return_mae=return_mae
                )
                batch_lwr[low_idx] = disc[0]
                batch_upr[low_idx] = disc[1]
                if return_mae:
                    batch_mae[low_idx] = disc[2]

        return batch_lwr, batch_upr, batch_mae

    def _compute_discrete_intervals_fallback(
        self,
        leaf_sub,
        active_weighting,
        n_neighbors,
        alpha_lwr,
        alpha_upr,
        return_mae=False,
    ):
        """Co-occurrence top-k interval (k = n_neighbors, or 25 for "auto"/"all") for low-support rows."""
        prox_disc = self._proximity_batch(leaf_sub, active_weighting, topological=False)

        if isinstance(n_neighbors, (int, np.integer)):
            k = min(int(n_neighbors), self.n_train)
        else:
            k = min(25, self.n_train)
        k_residuals = self._top_k_residuals(prox_disc, k)

        lwr = self.xp.quantile(k_residuals, alpha_lwr, axis=1)
        upr = self.xp.quantile(k_residuals, alpha_upr, axis=1)

        if return_mae:
            in_int = (k_residuals >= lwr[:, None]) & (k_residuals <= upr[:, None])
            cnt = self.xp.sum(in_int, axis=1)
            sum_abs = self.xp.sum(self.xp.abs(k_residuals) * in_int, axis=1)
            mae_val = self.xp.where(cnt > 0, sum_abs / self.xp.maximum(cnt, 1), float(self.oob_mae))
            return lwr, upr, mae_val
        return lwr, upr

    def compute_uq(
        self,
        X_test,
        n_neighbors="auto",
        level=0.95,
        use_density_scaling=None,
        weighting: str | None = None,
        use_leaf_weights: bool | None = None,
        weighted: bool | None = None,
        fallback_on_low_support: bool | None = None,
    ):
        """Local uncertainty: residual interval width at ``level``, expressed as a Gaussian sigma.

        Parameters
        ----------
        X_test : array-like of shape (n_test, n_features)
        n_neighbors : int, "auto" or "all", default="auto"
            Neighbourhood used for the residual quantiles (see :meth:`_interval_batch`).
        level : float, default=0.95
            Interval coverage.
        use_density_scaling : bool or None, default=None
            Override of the constructor setting.
        weighting, use_leaf_weights, weighted : optional
            Per-call overrides of the proximity weighting.
        fallback_on_low_support : bool or None, default=None
            Override of the constructor setting.

        Returns
        -------
        np.ndarray of shape (n_test,)
            ``gamma(x) * (q_upr - q_lwr) / (2 * z_{(1+level)/2})``.
        """
        if use_density_scaling is None:
            use_density_scaling = self.use_density_scaling

        active_weighting = self._resolve_runtime_weighting(weighting, use_leaf_weights, weighted)
        effective_fallback = (
            self.fallback_on_low_support if fallback_on_low_support is None else bool(fallback_on_low_support)
        )

        if not hasattr(self, "estimators"):
            self.fit()

        X_test = np.asarray(X_test)
        n_test = len(X_test)
        n_neighbors = self._parse_n_neighbors(n_neighbors, warn_on_float=True)

        leaf_matrix_test_xp = self._test_leaves(X_test)

        resid_lwr = self.xp.zeros(n_test, dtype=self.xp.float32)
        resid_upr = self.xp.zeros(n_test, dtype=self.xp.float32)

        alpha_lwr = (1.0 - level) / 2.0
        alpha_upr = 1.0 - alpha_lwr

        batch_size = self._resolve_batch_size(n_test)

        walked_density = (
            use_density_scaling
            and self.topological_decay_lambda is not None
            and self.topological_decay_lambda > 0.0
        )
        if walked_density:
            walked_densities = self.xp.zeros(n_test, dtype=self.xp.float32)

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            leaf_batch = leaf_matrix_test_xp[start:end, :]

            density_batch = self.xp.zeros(end - start, dtype=self.xp.float32) if walked_density else None
            prox_batch = self._proximity_batch(leaf_batch, active_weighting, density_batch=density_batch)
            if walked_density:
                walked_densities[start:end] = density_batch / self.n_estimators

            batch_lwr, batch_upr, _ = self._interval_batch(
                leaf_batch,
                prox_batch,
                active_weighting,
                n_neighbors,
                alpha_lwr,
                alpha_upr,
                return_mae=False,
                fallback=effective_fallback,
            )
            resid_lwr[start:end] = batch_lwr
            resid_upr[start:end] = batch_upr

        uq = resid_upr - resid_lwr

        if use_density_scaling:
            if walked_density:
                avg_test_leaf_sizes = walked_densities
            else:
                test_leaf_sizes = self.xp.zeros((n_test, self.n_estimators), dtype=self.xp.float32)
                for t in range(self.n_estimators):
                    test_leaf_sizes[:, t] = self.leaf_sizes[t][leaf_matrix_test_xp[:, t]]
                avg_test_leaf_sizes = self.xp.mean(test_leaf_sizes, axis=1)

            avg_test_leaf_sizes = self.xp.maximum(avg_test_leaf_sizes, 1e-5)
            gamma = self.xp.maximum(
                1.0, (self.N_baseline / avg_test_leaf_sizes) ** self.density_scaling_alpha
            )
            uq = uq * gamma

        # A central `level` interval of N(mu, sigma^2) spans 2 * z_{(1+level)/2} sigma.
        normal_divisor = 2.0 * float(norm.ppf((1.0 + level) / 2.0))
        uq = uq / normal_divisor

        if self.using_gpu:
            uq = uq.get()

        return uq

    def compute_density_factor(self, X_test: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Topological density factor ``gamma(x) = max(1, (N_baseline / n_walk(x))^alpha)``.

        ``n_walk(x)`` is the decay-weighted in-bag population around ``x`` averaged over trees.
        Requires ``topological_decay_lambda > 0``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "The UQ wrapper must be fitted via .fit() before calling .compute_density_factor()."
            )
        if self.topological_decay_lambda is None or self.topological_decay_lambda <= 0.0:
            raise ValueError("compute_density_factor requires topological_decay_lambda > 0.")

        X_test = np.asarray(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test[np.newaxis, :]
        n_test = len(X_test)
        alpha_val = self.density_scaling_alpha if alpha is None else float(alpha)

        leaf_matrix_test = self.model.apply(X_test).astype(np.int32, copy=False)
        leaf_matrix_test_xp = cp.asarray(leaf_matrix_test) if self.using_gpu else leaf_matrix_test

        batch_size = self._resolve_batch_size(n_test)
        dens = self.xp.zeros(n_test, dtype=self.xp.float32)

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            leaf_batch = leaf_matrix_test_xp[start:end, :]
            dens_batch = self.xp.zeros(end - start, dtype=self.xp.float32)
            for t in range(self.n_estimators):
                id_to_dense = self.tree_leaf_id_to_dense[t]
                dense_test = id_to_dense[leaf_batch[:, t]]
                dense_train = id_to_dense[self.in_bag_leaves_xp[:, t]]
                d_t = self.tree_leaf_distances[t][dense_test[:, None], dense_train[None, :]]
                decay = self.xp.exp(-self.topological_decay_lambda * d_t)
                dens_batch += self.xp.sum(decay * self.in_bag_counts_xp[None, :, t], axis=1)
            dens[start:end] = dens_batch

        dens = self.xp.maximum(dens / self.n_estimators, 1e-5)
        gamma = self.xp.maximum(1.0, (self.N_baseline / dens) ** alpha_val)
        return gamma.get() if self.using_gpu else np.asarray(gamma)

    @property
    def oob_mae(self) -> float:
        """Mean absolute training residual (OOB or CV)."""
        if hasattr(self, "oob_residuals") and self.oob_residuals is not None:
            resids = self.oob_residuals
            if hasattr(resids, "get"):
                resids = resids.get()
            return float(np.mean(np.abs(resids)))
        return 1.0

    def predict_with_intervals(
        self,
        X_test: np.ndarray,
        n_neighbors: int | str = "auto",
        level: float = 0.95,
        return_mae: bool = False,
        weighting: str | None = None,
        use_leaf_weights: bool | None = None,
        weighted: bool | None = None,
        fallback_on_low_support: bool | None = None,
    ):
        """Forest prediction with a proximity-based residual interval.

        Parameters are as in :meth:`compute_uq`; density scaling is not applied.

        Returns
        -------
        y_lwr, y_pred, y_upr : np.ndarray of shape (n_test,)
        local_mae : np.ndarray of shape (n_test,)
            Only if ``return_mae``: proximity-weighted mean absolute residual inside the interval.
        """
        if not hasattr(self, "estimators"):
            self.fit()

        active_weighting = self._resolve_runtime_weighting(weighting, use_leaf_weights, weighted)
        effective_fallback = (
            self.fallback_on_low_support if fallback_on_low_support is None else bool(fallback_on_low_support)
        )

        X_test = np.asarray(X_test)
        n_test = len(X_test)
        n_neighbors = self._parse_n_neighbors(n_neighbors, warn_on_float=False)

        leaf_matrix_test_xp = self._test_leaves(X_test)

        resid_lwr = self.xp.zeros(n_test, dtype=self.xp.float32)
        resid_upr = self.xp.zeros(n_test, dtype=self.xp.float32)
        local_mae = self.xp.zeros(n_test, dtype=self.xp.float32)

        alpha_lwr = (1.0 - level) / 2.0
        alpha_upr = 1.0 - alpha_lwr

        batch_size = self._resolve_batch_size(n_test)

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            leaf_batch = leaf_matrix_test_xp[start:end, :]
            prox_batch = self._proximity_batch(leaf_batch, active_weighting)
            batch_lwr, batch_upr, batch_mae = self._interval_batch(
                leaf_batch,
                prox_batch,
                active_weighting,
                n_neighbors,
                alpha_lwr,
                alpha_upr,
                return_mae=return_mae,
                fallback=effective_fallback,
            )
            resid_lwr[start:end] = batch_lwr
            resid_upr[start:end] = batch_upr
            if return_mae:
                local_mae[start:end] = batch_mae

        y_pred = self.model.predict(X_test)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        y_pred = np.asarray(y_pred, dtype=np.float32).flatten()

        if self.using_gpu:
            resid_lwr = resid_lwr.get().flatten()
            resid_upr = resid_upr.get().flatten()
            local_mae = local_mae.get().flatten()
        else:
            resid_lwr = np.asarray(resid_lwr, dtype=np.float32).flatten()
            resid_upr = np.asarray(resid_upr, dtype=np.float32).flatten()
            local_mae = np.asarray(local_mae, dtype=np.float32).flatten()

        y_pred_lwr = y_pred + resid_lwr
        y_pred_upr = y_pred + resid_upr

        if return_mae:
            return y_pred_lwr, y_pred, y_pred_upr, local_mae
        return y_pred_lwr, y_pred, y_pred_upr

    def _precompute_tree_paths(self, tree_idx):
        """Root-to-node paths (padded with -1) and depths of every node of one tree."""
        if not hasattr(self, "tree_paths"):
            self.tree_paths = {}
            self.tree_depths = {}

        tree = self.estimators[tree_idx].tree_
        node_count = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right

        parent = np.full(node_count, -1, dtype=np.int32)
        depth = np.zeros(node_count, dtype=np.int32)

        stack = [(0, 0)]
        while stack:
            node_id, curr_depth = stack.pop()
            depth[node_id] = curr_depth
            left = children_left[node_id]
            right = children_right[node_id]
            if left != -1:
                parent[left] = node_id
                stack.append((left, curr_depth + 1))
            if right != -1:
                parent[right] = node_id
                stack.append((right, curr_depth + 1))

        max_depth = np.max(depth) + 1

        node_paths = np.full((node_count, max_depth), -1, dtype=np.int32)
        for n in range(node_count):
            curr = n
            d = depth[n]
            while curr != -1:
                node_paths[n, d] = curr
                curr = parent[curr]
                d -= 1

        self.tree_paths[tree_idx] = self.xp.asarray(node_paths)
        self.tree_depths[tree_idx] = self.xp.asarray(depth)

    def compute_tree_topological_distances(self, leaf_test, leaf_train, tree_idx):
        """Path length between every pair of ``leaf_test`` and ``leaf_train`` nodes in one tree.

        ``d(i, j) = depth_i + depth_j - 2 * depth(LCA(i, j))``, computed in chunks of the test axis
        to bound the (chunk, n_train, max_depth) intermediate tensor.
        """
        if not hasattr(self, "tree_paths") or tree_idx not in self.tree_paths:
            self._precompute_tree_paths(tree_idx)

        node_paths = self.tree_paths[tree_idx]
        node_depths = self.tree_depths[tree_idx]

        leaf_test_xp = self.xp.asarray(leaf_test)
        leaf_train_xp = self.xp.asarray(leaf_train)

        n_test = len(leaf_test_xp)
        n_train = len(leaf_train_xp)

        max_elements = 3_000_000
        max_depth = node_paths.shape[1]
        batch_size = max(1, max_elements // (n_train * max_depth))

        d_t = self.xp.zeros((n_test, n_train), dtype=self.xp.float32)

        paths_train = node_paths[leaf_train_xp]
        train_depths = node_depths[leaf_train_xp][None, :]

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            sub_test = leaf_test_xp[start:end]
            paths_test = node_paths[sub_test]

            matches = paths_test[:, None, :] == paths_train[None, :, :]
            valid = (paths_test != -1)[:, None, :] & (paths_train != -1)[None, :, :]
            matches = matches & valid

            # The number of shared path prefix nodes is depth(LCA) + 1.
            lca_depth = self.xp.sum(matches, axis=2) - 1

            test_depths = node_depths[sub_test][:, None]
            d_t[start:end, :] = test_depths + train_depths - 2 * lca_depth

        return d_t

    def _compute_weighted_quantile(self, values, weights, q):
        """Row-wise weighted quantile (inverse empirical CDF) of ``values`` under ``weights``.

        Parameters
        ----------
        values : array of shape (n_train,)
        weights : array of shape (batch, n_train)
        q : float

        Returns
        -------
        array of shape (batch,)
            Rows with zero total weight get the unweighted quantile of ``values``.
        """
        sort_idx = self.xp.argsort(values)
        sorted_values = values[sort_idx]
        sorted_weights = weights[:, sort_idx]

        cum_weights = self.xp.cumsum(sorted_weights, axis=1)
        sum_weights = cum_weights[:, -1:]
        sum_weights_clipped = self.xp.maximum(sum_weights, 1e-10)
        cum_weights_norm = cum_weights / sum_weights_clipped

        idx_mask = cum_weights_norm >= q
        idx = self.xp.argmax(idx_mask, axis=1)
        val = sorted_values[idx]

        zero_weight_mask = sum_weights.ravel() <= 1e-10
        if self.xp.any(zero_weight_mask):
            fallback_val = self.xp.quantile(values, q)
            val = self.xp.where(zero_weight_mask, fallback_val, val)

        return val
