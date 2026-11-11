"""Entropy-based uncertainty decomposition of a random forest (Shaker & Hüllermeier, 2020).

Each tree is treated as a Gaussian component ``N(mu_t, sigma_t^2)`` built from its leaf statistics;
epistemic uncertainty is the mutual information ``H(mixture) - mean_t H(component_t)`` in bits.
"""

import warnings

import numpy as np
from scipy.special import logsumexp, roots_hermite

from dyrf_bo.uq.huber_entropy import huber_entropy_1d

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cp_logsumexp

    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            smoke_test = cp.asarray([1.0])
            smoke_test = smoke_test + 1.0
            cp.cuda.Stream.null.synchronize()
            HAS_GPU = bool(cp.asnumpy(smoke_test)[0] == 2.0)
        else:
            HAS_GPU = False
    except Exception:
        HAS_GPU = False
except ImportError:
    cp = None
    cp_logsumexp = None
    HAS_GPU = False


def compute_leaf_stats(estimator, test_leaf_ids, X_train=None, rf_parent=None, tree_idx=None, min_var=1e-6):
    """Unbiased leaf variances and effective sample counts for the given leaf ids.

    With ``X_train``, the tree's bootstrap multiplicities ``w_i`` are reconstructed and the
    leaf impurity is corrected with the weighted-sample factor ``V1^2 / (V1^2 - V2)``
    (``V1 = sum w_i``, ``V2 = sum w_i^2`` over the leaf); the effective count is Kish's
    ``V1^2 / V2``. Without ``X_train`` the plain Bessel correction ``n / (n - 1)`` is used.

    Parameters
    ----------
    estimator : DecisionTreeRegressor
        A fitted tree of the forest.
    test_leaf_ids : np.ndarray of shape (n_test,)
        Leaf index of each query point in ``estimator``.
    X_train : np.ndarray, optional
        Training inputs of the forest.
    rf_parent : RandomForestRegressor, optional
        Parent forest, used to read ``estimators_samples_``.
    tree_idx : int, optional
        Index of ``estimator`` within ``rf_parent``.
    min_var : float, default=1e-6
        Variance floor added to every leaf variance.

    Returns
    -------
    variances, counts : np.ndarray of shape (n_test,)
    """
    if not hasattr(estimator, "tree_"):
        return np.zeros(len(test_leaf_ids), dtype=np.float64), np.zeros(len(test_leaf_ids), dtype=np.float64)

    node_count = getattr(estimator.tree_, "node_count", len(estimator.tree_.impurity))

    if X_train is not None and len(X_train) > 0 and hasattr(estimator, "apply"):
        n_train = len(X_train)
        sample_indices = None
        if rf_parent is not None and hasattr(rf_parent, "estimators_samples_") and tree_idx is not None:
            if tree_idx < len(rf_parent.estimators_samples_):
                sample_indices = rf_parent.estimators_samples_[tree_idx]
        if sample_indices is None:
            is_bootstrap = getattr(rf_parent, "bootstrap", getattr(estimator, "bootstrap", True))
            if is_bootstrap:
                from sklearn.ensemble._forest import _generate_sample_indices

                sample_indices = _generate_sample_indices(estimator.random_state, n_train, n_train)
            else:
                sample_indices = np.arange(n_train)

        w = np.bincount(sample_indices, minlength=n_train)
        train_leaves = estimator.apply(X_train)
        V1 = np.bincount(train_leaves, weights=w, minlength=node_count)
        V2 = np.bincount(train_leaves, weights=w**2, minlength=node_count)

        denom = V1**2 - V2
        scale = np.zeros(node_count, dtype=np.float64)
        valid = denom > 0
        scale[valid] = (V1[valid] ** 2) / denom[valid]

        node_impurities = estimator.tree_.impurity
        node_variances = node_impurities * scale + min_var
        node_counts = np.divide(
            V1**2, V2, out=estimator.tree_.n_node_samples.astype(np.float64), where=V2 > 0
        )
    else:
        n_node_samples = estimator.tree_.n_node_samples.astype(np.float64)
        denom = np.maximum(n_node_samples - 1.0, 1.0)
        scale = np.where(n_node_samples > 1, n_node_samples / denom, 0.0)
        node_impurities = estimator.tree_.impurity
        node_variances = node_impurities * scale + min_var
        node_counts = n_node_samples

    return node_variances[test_leaf_ids], node_counts[test_leaf_ids]


class EpistemicQuantifier:
    """Per-tree Gaussian view of a fitted random forest.

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted forest.
    X_train, y_train : array-like
        Training data of ``model``; ``X_train`` is needed to reconstruct bootstrap weights.
    """

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = np.asarray(X_train) if X_train is not None else None
        self.y_train = np.asarray(y_train) if y_train is not None else None

    def _base_calc_per_tree_variance(self, X_test, min_var=1e-6, all_test_leaf_ids=None):
        """Unbiased leaf variance of every tree at every query point, shape (n_trees, n_test)."""
        X_test = np.atleast_2d(X_test)
        n_trees = len(self.model.estimators_)
        n_samples = X_test.shape[0]

        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)

        variances = np.zeros((n_trees, n_samples))
        for t, estimator in enumerate(self.model.estimators_):
            test_leaf_ids = all_test_leaf_ids[:, t]
            v_t, _ = compute_leaf_stats(
                estimator,
                test_leaf_ids,
                X_train=self.X_train,
                rf_parent=self.model,
                tree_idx=t,
                min_var=min_var,
            )
            variances[t, :] = v_t

        return variances

    def _get_tree_predictions(self, X_test, all_test_leaf_ids=None):
        """Prediction of every tree at every query point, shape (n_trees, n_test)."""
        X_test = np.atleast_2d(X_test)
        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)
        n_samples = X_test.shape[0]
        n_trees = len(self.model.estimators_)

        tree_preds = np.zeros((n_trees, n_samples))
        for t, estimator in enumerate(self.model.estimators_):
            tree_preds[t, :] = estimator.tree_.value[all_test_leaf_ids[:, t], 0, 0]
        return tree_preds

    def base_get_aleatoric_variance(self, X_test, all_test_leaf_ids=None):
        """Aleatoric variance ``E_t[sigma_t^2]`` at each query point."""
        return np.mean(self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids), axis=0)

    def standard_get_epistemic_variance(self, X_test):
        """Variance of the tree predictions, ``E[h^2] - E[h]^2``, clipped at zero."""
        tree_preds = self._get_tree_predictions(X_test)
        mean_of_squares = np.mean(tree_preds**2, axis=0)
        square_of_mean = (np.mean(tree_preds, axis=0)) ** 2
        variance = mean_of_squares - square_of_mean
        return np.maximum(variance, 0.0)

    def shaker_get_epistemic_entropy(
        self,
        X_test,
        num_samples=10000,
        batch_size="auto",
        random_state=None,
        backend="auto",
        n_quadrature_points=32,
        method="huber",
        enable_splitting=False,
    ):
        """Epistemic uncertainty as mutual information ``H_total - H_aleatoric`` in bits.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features) or (n_features,)
            Query points.
        num_samples : int, default=10000
            Monte Carlo samples per point (``method="monte_carlo"``).
        batch_size : int or "auto", default="auto"
            Number of query points processed at once.
        random_state : int, Generator or None, default=None
            Seed for ``method="monte_carlo"``.
        backend : {"auto", "cpu", "gpu"}, default="auto"
            Computation backend.
        n_quadrature_points : int, default=32
            Gauss-Hermite nodes per component (``method="gauss_hermite"``).
        method : {"huber", "gauss_hermite", "monte_carlo"}, default="huber"
            Estimator of the mixture entropy; "huber" is the closed-form Huber et al. (2008) H_2.
        enable_splitting : bool, default=False
            Split each component with Huber's Table I library before the Taylor expansion.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Mutual information clipped to ``[0, log2(n_trees)]``.
        """
        X_test = np.atleast_2d(X_test)
        all_test_leaf_ids = self.model.apply(X_test)
        n_trees = len(self.model.estimators_)
        max_mi_bound = float(np.log2(n_trees))

        total_unc = self._shaker_calc_total_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
            all_test_leaf_ids=all_test_leaf_ids,
            n_quadrature_points=n_quadrature_points,
            method=method,
            enable_splitting=enable_splitting,
        )
        aleatoric_unc = self._shaker_calc_aleatoric_entropy(X_test, all_test_leaf_ids=all_test_leaf_ids)

        raw_mi = total_unc - aleatoric_unc
        if np.any(raw_mi > max_mi_bound + 1e-4):
            max_viol = float(np.max(raw_mi - max_mi_bound))
            warnings.warn(
                f"Numerical integration exceeded information-theoretic MI upper bound "
                f"log2({n_trees})={max_mi_bound:.4f} by {max_viol:.4f} bits. Clamping to bound.",
                RuntimeWarning,
                stacklevel=1,
            )

        return np.clip(raw_mi, 0.0, max_mi_bound)

    def shaker_get_epistemic_variance(
        self,
        X_test,
        num_samples=10000,
        batch_size="auto",
        random_state=None,
        backend="auto",
        n_quadrature_points=32,
        method="huber",
        enable_splitting=False,
    ):
        """Mutual information mapped to an epistemic variance.

        Inverts ``MI = 0.5 * log2(total_var / aleatoric_var)`` to
        ``epistemic_var = aleatoric_var * (2 ** (2 * MI) - 1)``.
        Parameters are those of :meth:`shaker_get_epistemic_entropy`.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        X_test = np.atleast_2d(X_test)
        all_test_leaf_ids = self.model.apply(X_test)

        mi = self.shaker_get_epistemic_entropy(
            X_test,
            num_samples=num_samples,
            batch_size=batch_size,
            random_state=random_state,
            backend=backend,
            n_quadrature_points=n_quadrature_points,
            method=method,
            enable_splitting=enable_splitting,
        )
        aleatoric_var = self.base_get_aleatoric_variance(X_test, all_test_leaf_ids=all_test_leaf_ids)

        # Clip the exponent so 2**x cannot overflow float64.
        safe_exponent = np.clip(2.0 * mi, 0.0, 50.0)
        return aleatoric_var * np.maximum(2.0**safe_exponent - 1.0, 0.0)

    def _shaker_calc_aleatoric_entropy(self, X_test, all_test_leaf_ids=None):
        """Mean differential entropy of the tree components, ``mean_t 0.5 * log2(2 pi e sigma_t^2)``."""
        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)

        vars2 = self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids)
        individual_entropies = 0.5 * np.log2(2 * np.pi * np.e * vars2)
        return np.mean(individual_entropies, axis=0)

    def _shaker_calc_total_entropy(
        self,
        X_test,
        num_samples=10000,
        batch_size="auto",
        random_state=None,
        backend="auto",
        all_test_leaf_ids=None,
        n_quadrature_points=32,
        method="huber",
        enable_splitting=False,
    ):
        """Differential entropy (bits) of the equally weighted mixture of tree Gaussians.

        See :meth:`shaker_get_epistemic_entropy` for the parameters.
        """
        X_test = np.atleast_2d(X_test)
        n_samples = X_test.shape[0]
        n_trees = len(self.model.estimators_)

        backend = self._mc_resolve_backend(backend)
        is_gpu = backend == "gpu"

        if all_test_leaf_ids is None:
            all_test_leaf_ids = self.model.apply(X_test)

        mu_all = self._get_tree_predictions(X_test, all_test_leaf_ids=all_test_leaf_ids)
        vars_all = self._base_calc_per_tree_variance(X_test, all_test_leaf_ids=all_test_leaf_ids)

        vars_all = np.maximum(vars_all, 1e-6)
        sigmas_all = np.sqrt(vars_all)

        if method == "huber":
            return huber_entropy_1d(
                means=mu_all.T,
                variances=sigmas_all.T**2,
                weights=None,
                enable_splitting=enable_splitting,
                backend=backend,
                batch_size=batch_size,
            )

        total_entropy = np.zeros(n_samples)

        if method == "monte_carlo":
            rng = np.random.default_rng(random_state)
            mc_batch_size = 50 if batch_size == "auto" else batch_size
            for start in range(0, n_samples, mc_batch_size):
                end = min(start + mc_batch_size, n_samples)
                B = end - start
                mu_batch = mu_all[:, start:end]
                sigma_batch = sigmas_all[:, start:end]

                tree_idx = rng.integers(0, n_trees, size=(num_samples, B))
                m_chosen = np.take_along_axis(mu_batch, tree_idx, axis=0)
                s_chosen = np.take_along_axis(sigma_batch, tree_idx, axis=0)

                y_samples = m_chosen + s_chosen * rng.standard_normal(size=(num_samples, B))

                u = (y_samples[:, None, :] - mu_batch[None, :, :]) / sigma_batch[None, :, :]
                log_comp = -0.5 * u**2 - np.log(sigma_batch[None, :, :]) - 0.5 * np.log(2.0 * np.pi)
                log_py = logsumexp(log_comp, axis=1) - np.log(n_trees)
                log2_py = log_py / np.log(2.0)

                total_entropy[start:end] = -np.mean(log2_py, axis=0)
            return total_entropy

        # Gauss-Hermite quadrature around each component; batches are halved on memory errors.
        K = n_quadrature_points
        z_gh, w_gh = roots_hermite(K)

        if batch_size == "auto":
            bytes_per_sample = n_trees * K * n_trees * 8
            batch_size = max(1, min(64, int(50_000_000 / (bytes_per_sample + 1))))

        if is_gpu and cp is not None and cp_logsumexp is not None:
            z_g = cp.asarray(z_gh)
            w_g = cp.asarray(w_gh)
            norm_w = (w_g / float(np.sqrt(np.pi)))[None, :, None]
            mu_g = cp.asarray(mu_all)
            sigmas_g = cp.asarray(sigmas_all)
            sqrt2 = float(np.sqrt(2.0))
            half_log_2pi = float(0.5 * np.log(2.0 * np.pi))
            log_ntrees = float(np.log(n_trees))
            ln2 = float(np.log(2.0))

            start = 0
            while start < n_samples:
                end = min(start + batch_size, n_samples)
                B = end - start
                try:
                    mu_b = mu_g[:, start:end]
                    sig_b = sigmas_g[:, start:end]

                    y = mu_b[:, None, :] + sqrt2 * sig_b[:, None, :] * z_g[None, :, None]
                    y_exp = y[:, :, None, :]
                    mu_exp = mu_b[None, None, :, :]
                    sig_exp = sig_b[None, None, :, :]

                    u = (y_exp - mu_exp) / sig_exp
                    log_comp = -0.5 * u**2 - cp.log(sig_exp) - half_log_2pi
                    log_py = cp_logsumexp(log_comp, axis=2) - log_ntrees
                    log2_py = log_py / ln2

                    h_per_tree = cp.sum(norm_w * (-log2_py), axis=1)
                    batch_entropy = cp.mean(h_per_tree, axis=0)
                    total_entropy[start:end] = cp.asnumpy(batch_entropy)
                    start += B
                except Exception:
                    cp.get_default_memory_pool().free_all_blocks()
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
        else:
            norm_w = (w_gh / np.sqrt(np.pi))[None, :, None]
            start = 0
            while start < n_samples:
                end = min(start + batch_size, n_samples)
                B = end - start
                try:
                    mu_b = mu_all[:, start:end]
                    sig_b = sigmas_all[:, start:end]

                    y = mu_b[:, None, :] + np.sqrt(2.0) * sig_b[:, None, :] * z_gh[None, :, None]
                    y_exp = y[:, :, None, :]
                    mu_exp = mu_b[None, None, :, :]
                    sig_exp = sig_b[None, None, :, :]

                    u = (y_exp - mu_exp) / sig_exp
                    log_comp = -0.5 * u**2 - np.log(sig_exp) - 0.5 * np.log(2.0 * np.pi)
                    log_py = logsumexp(log_comp, axis=2) - np.log(n_trees)
                    log2_py = log_py / np.log(2.0)

                    h_per_tree = np.sum(norm_w * (-log2_py), axis=1)
                    batch_entropy = np.mean(h_per_tree, axis=0)
                    total_entropy[start:end] = batch_entropy
                    start += B
                except Exception:
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)

        return total_entropy

    def _mc_resolve_backend(self, backend):
        """Map "auto"/"gpu" to "cpu" when no usable GPU is present."""
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be one of: 'auto', 'cpu', 'gpu'")
        if backend == "auto":
            return "gpu" if HAS_GPU else "cpu"
        if backend == "gpu" and not HAS_GPU:
            return "cpu"
        return backend
