"""Closed-form second-order entropy approximation H_2 for 1D Gaussian mixtures (NumPy backend).

Reference: Huber, Bailey, Durrant-Whyte and Hanebeck, "On Entropy Approximation for Gaussian
Mixture Random Vectors", IEEE MFI 2008, pp. 181-188.
"""

import warnings

import numpy as np

INV_LN2: float = float(1.0 / np.log(2.0))
HALF_LOG_2PI: float = float(0.5 * np.log(2.0 * np.pi))
EPSILON_LOG_WEIGHT: float = 1e-300
MIN_SAFE_VAR: float = 1e-15

# Huber (2008) Table I: four-component splitting library for N(0, 1).
_RAW_TABLE1_WEIGHTS = np.array([0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098], dtype=np.float64)
_RAW_TABLE1_WEIGHTS.flags.writeable = False

_RAW_TABLE1_MEANS = np.array([-1.41312052330, -0.44973059608, 0.44973059608, 1.41312052330], dtype=np.float64)
_RAW_TABLE1_MEANS.flags.writeable = False

# The tabulated weights sum to 1 - 2e-11; renormalise them exactly.
_NORM_TABLE1_WEIGHTS = _RAW_TABLE1_WEIGHTS / np.sum(_RAW_TABLE1_WEIGHTS)
_NORM_TABLE1_WEIGHTS.flags.writeable = False

# Instead of the tabulated sigma = 0.51751260421, use the common sigma that makes the split
# mixture have exactly unit variance: sum(w * (mu^2 + sigma^2)) == 1.
_TARGET_SIGMA2 = (1.0 - np.sum(_NORM_TABLE1_WEIGHTS * _RAW_TABLE1_MEANS**2)) / np.sum(_NORM_TABLE1_WEIGHTS)
_MOMENT_MATCHED_STDS = np.full(4, np.sqrt(_TARGET_SIGMA2), dtype=np.float64)
_MOMENT_MATCHED_STDS.flags.writeable = False

# Imported after the Table I constants: huber_entropy_gpu imports them from this module.
from dyrf_bo.uq.huber_entropy_gpu import (  # noqa: E402
    HAS_GPU,
    clear_gpu_memory_pool,
    get_dynamic_huber_batch_size_gpu,
    huber_entropy_1d_cupy,
)


def _validate_and_preprocess_inputs(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray | None = None,
    min_var: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate mixture parameters and standardise them to 2D float64 arrays of shape (B, T).

    Parameters
    ----------
    means, variances : np.ndarray of shape (B, T) or (T,)
        Component means and variances.
    weights : np.ndarray of shape (B, T), (1, T) or (T,), optional
        Component weights; uniform if None.
    min_var : float, default=1e-6
        Variance floor.

    Returns
    -------
    means_arr, vars_clipped, w_norm : np.ndarray of shape (B, T)
        Means, floored variances and normalised weights.
    B, T : int
        Batch size and number of components.

    Raises
    ------
    ValueError
        On non-finite values, negative variances or weights, or inconsistent shapes.
    """
    means_arr = np.asarray(means, dtype=np.float64)
    vars_arr = np.asarray(variances, dtype=np.float64)

    if not np.all(np.isfinite(means_arr)):
        raise ValueError("means must contain only finite real values (found NaN or Inf)")
    if not np.all(np.isfinite(vars_arr)):
        raise ValueError("variances must contain only finite real values (found NaN or Inf)")

    if np.any(vars_arr < 0.0):
        raise ValueError("Component variances must be non-negative (found negative variance < 0)")

    if means_arr.ndim == 1:
        means_arr = means_arr[np.newaxis, :]
    elif means_arr.ndim != 2:
        raise ValueError(f"means must be 1D or 2D array, got shape {means_arr.shape}")

    if vars_arr.ndim == 1:
        vars_arr = vars_arr[np.newaxis, :]
    elif vars_arr.ndim != 2:
        raise ValueError(f"variances must be 1D or 2D array, got shape {vars_arr.shape}")

    if means_arr.shape != vars_arr.shape:
        raise ValueError(f"Shape mismatch between means {means_arr.shape} and variances {vars_arr.shape}")

    B, T = means_arr.shape
    if T < 1:
        raise ValueError(f"Number of mixture components T must be at least 1, got {T}")

    safe_min_var = max(float(min_var), MIN_SAFE_VAR)
    vars_clipped = np.maximum(vars_arr, safe_min_var)

    if weights is None:
        w_norm = np.full((B, T), 1.0 / T, dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("weights must contain only finite real values (found NaN or Inf)")
        if np.any(w_arr < 0.0):
            raise ValueError("Component mixture weights must be non-negative")

        if w_arr.ndim == 1:
            if w_arr.shape[0] != T:
                raise ValueError(
                    f"weights 1D shape ({w_arr.shape[0]},) does not match number of components T={T}"
                )
            w_arr = np.tile(w_arr[np.newaxis, :], (B, 1))
        elif w_arr.ndim == 2:
            if w_arr.shape == (1, T) and B > 1:
                w_arr = np.tile(w_arr, (B, 1))
            elif w_arr.shape != (B, T):
                raise ValueError(f"weights 2D shape {w_arr.shape} does not match (B, T)=({B}, {T})")
        else:
            raise ValueError(f"weights must be 1D or 2D array, got shape {w_arr.shape}")

        w_sums = np.sum(w_arr, axis=-1, keepdims=True)
        if np.any(w_sums <= 0.0):
            raise ValueError("Component weights must sum to a strictly positive value")
        w_norm = w_arr / w_sums

    return means_arr, vars_clipped, w_norm, B, T


def _compute_h2_bits(
    means_arr: np.ndarray,
    vars_clipped: np.ndarray,
    w_norm: np.ndarray,
    delta: np.ndarray | None = None,
) -> np.ndarray:
    """Second-order Taylor entropy H_2 in bits, without splitting.

    Parameters
    ----------
    means_arr, vars_clipped, w_norm : np.ndarray of shape (B, T)
        Validated means, floored variances and normalised weights.
    delta : np.ndarray of shape (B, T, T), optional
        Pairwise differences ``means_arr[:, :, None] - means_arr[:, None, :]``.

    Returns
    -------
    np.ndarray of shape (B,)
    """
    if delta is None:
        delta = means_arr[:, :, np.newaxis] - means_arr[:, np.newaxis, :]  # (B, T, T)

    vars_j = vars_clipped[:, np.newaxis, :]  # (B, 1, T)
    inv_vars_j = 1.0 / vars_j
    u = delta * inv_vars_j  # (mu_i - mu_j) / sigma_j^2

    log_w = np.where(w_norm > 0.0, np.log(np.maximum(w_norm, EPSILON_LOG_WEIGHT)), -np.inf)
    log_w_j = log_w[:, np.newaxis, :]  # (B, 1, T)

    log_norm = -HALF_LOG_2PI - 0.5 * np.log(vars_j) - 0.5 * (delta * u)
    ell = log_w_j + log_norm  # (B, T, T)

    # Log-density and component responsibilities at each mean in one stable log-sum-exp pass.
    m = np.max(ell, axis=-1, keepdims=True)
    is_neg_inf = np.isneginf(m)
    safe_m = np.where(is_neg_inf, 0.0, m)
    tmp = np.exp(ell - safe_m)
    sum_tmp = np.sum(tmp, axis=-1, keepdims=True)
    safe_sum = np.maximum(sum_tmp, 1e-300)

    log_f = np.squeeze(np.where(is_neg_inf, -np.inf, np.log(safe_sum) + safe_m), axis=-1)  # (B, T)
    responsibilities = np.where(is_neg_inf, 0.0, tmp / safe_sum)  # (B, T, T)

    grad_ratio = -np.sum(responsibilities * u, axis=-1)  # (B, T)
    second_ratio = np.sum(responsibilities * (u * u - inv_vars_j), axis=-1)  # (B, T)
    f_hessian = second_ratio - (grad_ratio**2)  # (B, T)

    h0 = -np.sum(w_norm * log_f, axis=-1)  # (B,)
    h2_nats = h0 - 0.5 * np.sum(w_norm * f_hessian * vars_clipped, axis=-1)  # (B,)
    h2_bits = h2_nats * INV_LN2

    return h2_bits


def _compute_h2_split_bits(
    means_orig: np.ndarray,
    vars_orig: np.ndarray,
    w_orig: np.ndarray,
    means_split: np.ndarray,
    vars_split: np.ndarray,
    w_split: np.ndarray,
) -> np.ndarray:
    """Second-order Taylor entropy H_2 in bits with component splitting.

    Following Huber's Remark 1, the outer expectation runs over the 4T split components while the
    mixture log-density inside it is that of the original T components (cross tensor (B, 4T, T)).

    Parameters
    ----------
    means_orig, vars_orig, w_orig : np.ndarray of shape (B, T)
        Original mixture.
    means_split, vars_split, w_split : np.ndarray of shape (B, 4T)
        Split mixture from :func:`huber_table1_split_1d`.

    Returns
    -------
    np.ndarray of shape (B,)
    """
    delta = means_split[:, :, np.newaxis] - means_orig[:, np.newaxis, :]  # (B, 4T, T)

    vars_j = vars_orig[:, np.newaxis, :]  # (B, 1, T)
    inv_vars_j = 1.0 / vars_j  # (B, 1, T)
    u = delta * inv_vars_j  # (B, 4T, T)

    log_w = np.where(w_orig > 0.0, np.log(np.maximum(w_orig, EPSILON_LOG_WEIGHT)), -np.inf)
    log_w_j = log_w[:, np.newaxis, :]  # (B, 1, T)

    log_norm = -HALF_LOG_2PI - 0.5 * np.log(vars_j) - 0.5 * (delta * u)
    ell = log_w_j + log_norm  # (B, 4T, T)

    m = np.max(ell, axis=-1, keepdims=True)  # (B, 4T, 1)
    is_neg_inf = np.isneginf(m)
    safe_m = np.where(is_neg_inf, 0.0, m)
    tmp = np.exp(ell - safe_m)  # (B, 4T, T)
    sum_tmp = np.sum(tmp, axis=-1, keepdims=True)  # (B, 4T, 1)
    safe_sum = np.maximum(sum_tmp, 1e-300)

    log_g = np.squeeze(np.where(is_neg_inf, -np.inf, np.log(safe_sum) + safe_m), axis=-1)  # (B, 4T)
    responsibilities = np.where(is_neg_inf, 0.0, tmp / safe_sum)  # (B, 4T, T)

    grad_ratio = -np.sum(responsibilities * u, axis=-1)  # (B, 4T)
    second_ratio = np.sum(responsibilities * (u * u - inv_vars_j), axis=-1)  # (B, 4T)
    f_hessian = second_ratio - (grad_ratio**2)  # (B, 4T)

    h0 = -np.sum(w_split * log_g, axis=-1)  # (B,)
    h2_nats = h0 - 0.5 * np.sum(w_split * f_hessian * vars_split, axis=-1)  # (B,)
    h2_bits = h2_nats * INV_LN2

    return h2_bits


def huber_table1_split_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray | None = None,
    min_var: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every component into four using the (moment-matched) Huber Table I library.

    Parameters
    ----------
    means, variances : np.ndarray of shape (B, T) or (T,)
        Component means and variances.
    weights : np.ndarray, optional
        Component weights; uniform if None.
    min_var : float, default=1e-6
        Variance floor.

    Returns
    -------
    means_split, variances_split, weights_split : np.ndarray of shape (B, 4T)
    """
    means_arr, vars_clipped, w_norm, B, T = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        return (
            np.empty((0, 4 * T), dtype=np.float64),
            np.empty((0, 4 * T), dtype=np.float64),
            np.empty((0, 4 * T), dtype=np.float64),
        )

    w_lib = _NORM_TABLE1_WEIGHTS
    mu_lib = _RAW_TABLE1_MEANS
    std_lib = _MOMENT_MATCHED_STDS

    # mu_{b,t,k} = mu_{b,t} + sigma_{b,t} * mu_lib_k,  var_{b,t,k} = var_{b,t} * std_lib_k^2,
    # w_{b,t,k} = w_{b,t} * w_lib_k
    std_orig = np.sqrt(vars_clipped)  # (B, T)
    means_expanded = (
        means_arr[:, :, np.newaxis] + std_orig[:, :, np.newaxis] * mu_lib[np.newaxis, np.newaxis, :]
    )  # (B, T, 4)
    vars_expanded = vars_clipped[:, :, np.newaxis] * (std_lib**2)[np.newaxis, np.newaxis, :]  # (B, T, 4)
    weights_expanded = w_norm[:, :, np.newaxis] * w_lib[np.newaxis, np.newaxis, :]  # (B, T, 4)

    means_split = means_expanded.reshape(B, 4 * T)
    vars_split = vars_expanded.reshape(B, 4 * T)
    weights_split = weights_expanded.reshape(B, 4 * T)

    return means_split, vars_split, weights_split


def huber_entropy_1d_numpy(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray | None = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
) -> np.ndarray:
    """Second-order Huber (2008) differential entropy H_2 of 1D Gaussian mixtures, in bits.

    Parameters
    ----------
    means, variances : np.ndarray of shape (B, T) or (T,)
        Component means and variances.
    weights : np.ndarray of shape (B, T), (1, T) or (T,), optional
        Component weights; uniform if None.
    min_var : float, default=1e-6
        Variance floor.
    enable_splitting : bool, default=False
        Split each component with :func:`huber_table1_split_1d` before the Taylor expansion.

    Returns
    -------
    np.ndarray of shape (B,)

    Raises
    ------
    ValueError
        On invalid inputs (see :func:`_validate_and_preprocess_inputs`).
    """
    means_arr, vars_clipped, w_norm, B, T = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        return np.empty((0,), dtype=np.float64)

    if enable_splitting:
        means_sp, vars_sp, w_sp = huber_table1_split_1d(
            means_arr, vars_clipped, weights=w_norm, min_var=min_var
        )
        return _compute_h2_split_bits(means_arr, vars_clipped, w_norm, means_sp, vars_sp, w_sp)

    delta = means_arr[:, :, np.newaxis] - means_arr[:, np.newaxis, :]
    return _compute_h2_bits(means_arr, vars_clipped, w_norm, delta=delta)


def _execute_chunk_gpu(
    b_means: np.ndarray,
    b_vars: np.ndarray,
    b_weights: np.ndarray,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
) -> np.ndarray:
    """Evaluate one chunk on the GPU, halving the sub-chunk size on out-of-memory errors.

    Falls back to :func:`huber_entropy_1d_numpy` when CuPy is missing or a single row does not fit.
    """
    from dyrf_bo.uq import huber_entropy_gpu

    current_cp = getattr(huber_entropy_gpu, "cp", None)
    if current_cp is None:
        warnings.warn(
            "GPU backend requested but CuPy module not found. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return huber_entropy_1d_numpy(
            b_means, b_vars, weights=b_weights, min_var=min_var, enable_splitting=enable_splitting
        )

    chunk_len = b_means.shape[0]
    out_dtype = b_means.dtype if b_means.dtype in (np.float32, np.float64) else np.float64
    if chunk_len == 0:
        return np.empty((0,), dtype=out_dtype)

    sub_chunk_size = chunk_len

    while True:
        try:
            if sub_chunk_size >= chunk_len:
                res = huber_entropy_1d_cupy(
                    current_cp.asarray(b_means),
                    current_cp.asarray(b_vars),
                    weights_cp=current_cp.asarray(b_weights),
                    min_var=min_var,
                    enable_splitting=enable_splitting,
                )
                return current_cp.asnumpy(res)

            sub_h2 = np.empty((chunk_len,), dtype=out_dtype)
            for sub_start in range(0, chunk_len, sub_chunk_size):
                sub_end = min(sub_start + sub_chunk_size, chunk_len)
                sub_res = huber_entropy_1d_cupy(
                    current_cp.asarray(b_means[sub_start:sub_end]),
                    current_cp.asarray(b_vars[sub_start:sub_end]),
                    weights_cp=current_cp.asarray(b_weights[sub_start:sub_end]),
                    min_var=min_var,
                    enable_splitting=enable_splitting,
                )
                sub_h2[sub_start:sub_end] = current_cp.asnumpy(sub_res)
            return sub_h2

        except Exception as e:
            is_oom = "out of memory" in str(e).lower() or isinstance(e, MemoryError)
            if hasattr(current_cp, "cuda") and hasattr(current_cp.cuda, "memory"):
                if isinstance(e, getattr(current_cp.cuda.memory, "OutOfMemoryError", ())):
                    is_oom = True

            if not is_oom:
                raise
            clear_gpu_memory_pool()
            sub_chunk_size = sub_chunk_size // 2
            if sub_chunk_size < 1:
                warnings.warn(
                    "GPU out of memory during Huber entropy computation. Falling back to CPU for this batch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return huber_entropy_1d_numpy(
                    b_means, b_vars, weights=b_weights, min_var=min_var, enable_splitting=enable_splitting
                )


def huber_entropy_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray | None = None,
    min_var: float = 1e-6,
    enable_splitting: bool = False,
    backend: str = "auto",
    batch_size: str | int = "auto",
) -> np.ndarray:
    """Huber (2008) H_2 entropy in bits with CPU/GPU dispatch and chunking over the batch axis.

    Parameters
    ----------
    means, variances : np.ndarray of shape (B, T) or (T,)
        Component means and variances.
    weights : np.ndarray of shape (B, T), (1, T) or (T,), optional
        Component weights; uniform if None.
    min_var : float, default=1e-6
        Variance floor.
    enable_splitting : bool, default=False
        Split each component with Huber's Table I library before the Taylor expansion.
    backend : {"auto", "cpu", "gpu"}, default="auto"
        "auto" uses the GPU when CuPy and a CUDA device are available; "gpu" without a device
        warns and runs on the CPU.
    batch_size : int or "auto", default="auto"
        Rows per chunk; "auto" derives it from free VRAM (GPU) or caps it at 5000 (CPU).

    Returns
    -------
    np.ndarray of shape (B,)

    Raises
    ------
    ValueError
        On an unknown backend, an invalid batch size or invalid mixture parameters.
    """
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"backend must be one of: 'auto', 'cpu', 'gpu', got {backend!r}")

    if isinstance(batch_size, str):
        if batch_size != "auto":
            raise ValueError(f"batch_size must be 'auto' or a positive integer, got {batch_size!r}")
    elif isinstance(batch_size, (int, np.integer)):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be 'auto' or a positive integer, got {batch_size!r}")
    else:
        raise ValueError(f"batch_size must be 'auto' or a positive integer, got {type(batch_size).__name__}")

    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "gpu" if HAS_GPU else "cpu"
    elif backend == "gpu" and not HAS_GPU:
        warnings.warn(
            "GPU backend requested but CUDA/CuPy is unavailable. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        resolved_backend = "cpu"

    means_arr, vars_clipped, w_norm, B, T = _validate_and_preprocess_inputs(
        means, variances, weights=weights, min_var=min_var
    )

    if B == 0:
        return np.empty((0,), dtype=np.float64)

    if isinstance(batch_size, (int, np.integer)):
        chunk_size = int(batch_size)
    elif resolved_backend == "gpu":
        chunk_size = get_dynamic_huber_batch_size_gpu(
            T, enable_splitting=enable_splitting, dtype=means_arr.dtype
        )
    else:
        chunk_size = min(B, 5000)

    def run_chunk(b_means, b_vars, b_weights):
        if resolved_backend == "gpu":
            return _execute_chunk_gpu(
                b_means, b_vars, b_weights, min_var=min_var, enable_splitting=enable_splitting
            )
        return huber_entropy_1d_numpy(
            b_means, b_vars, weights=b_weights, min_var=min_var, enable_splitting=enable_splitting
        )

    if B <= chunk_size:
        return run_chunk(means_arr, vars_clipped, w_norm)

    out_dtype = means_arr.dtype if means_arr.dtype in (np.float32, np.float64) else np.float64
    h_2_all = np.empty((B,), dtype=out_dtype)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        h_2_all[start:end] = run_chunk(means_arr[start:end], vars_clipped[start:end], w_norm[start:end])
    return h_2_all
