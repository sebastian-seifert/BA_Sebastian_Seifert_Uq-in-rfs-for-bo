"""Single extrapolation UQ experiment: sampling, surrogate fit, per-point table and scorecard."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .convex_hull import ConvexHullProjectionSolver
from .metrics import compute_comprehensive_metrics
from .objectives import get_objective
from .samplers import (
    sample_natural_test,
    sample_stratified_test,
    sample_subdomain_training,
)
from .uq_evaluator import APPROACH_NAMES, MultiUQEvaluator


@dataclass
class ExtrapolationRunConfig:
    """Configuration of one extrapolation experiment.

    Attributes
    ----------
    dimension : int
        Input dimensionality D.
    n_train : int
        Number of training points in the sub-domain [-0.5, 0.5]^D.
    function_name : str
        Objective name ('sphere', 'rosenbrock', 'rastrigin', 'ackley').
    sampling_strategy : str, default="stratified"
        Test sampling protocol ('natural' or 'stratified').
    seed : int, default=0
        Master seed; independent train, test and model seeds are spawned from it.
    n_test : int, default=10000
        Total number of test points.
    k : int, default=28
        Neighbourhood size of proximity method A.
    eps : float, default=0.080791
        Exploration floor coefficient epsilon.
    decay_lambda : float, default=0.20486
        Topological decay rate lambda.
    n_trees : int, default=10
        Number of trees (10 selects the surrogate's default).
    surrogate_type : str, default="smac_default"
        Surrogate configuration name.
    eval_mode : str, default="all"
        Passed to ``MultiUQEvaluator``.
    """

    dimension: int
    n_train: int
    function_name: str
    sampling_strategy: str = "stratified"
    seed: int = 0
    n_test: int = 10000
    k: int = 28
    eps: float = 0.080791
    decay_lambda: float = 0.20486
    n_trees: int = 10
    surrogate_type: str = "smac_default"
    eval_mode: str = "all"


def run_single_experiment(
    config: ExtrapolationRunConfig,
    output_dir: str | Path | None = None,
    save_parquet: bool = True,
    compute_metrics: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one extrapolation experiment.

    Parameters
    ----------
    config : ExtrapolationRunConfig
        Experiment configuration.
    output_dir : str, Path or None, default=None
        Directory for the per-point parquet file, or a ``.parquet`` file path.
    save_parquet : bool, default=True
        Whether to write the per-point table.
    compute_metrics : bool, default=True
        Whether to compute the calibration scorecard.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        Summary dictionary and per-point DataFrame.
    """
    if config.sampling_strategy not in {"natural", "stratified"}:
        raise ValueError(
            f"Unsupported sampling strategy '{config.sampling_strategy}'. Must be 'natural' or 'stratified'."
        )

    objective = get_objective(config.function_name)

    # Independent streams for training data, test data and the model.
    ss = np.random.SeedSequence(config.seed)
    seed_train, seed_test, seed_model = [int(s.generate_state(1)[0]) for s in ss.spawn(3)]

    X_train = sample_subdomain_training(
        n_samples=config.n_train,
        dimension=config.dimension,
        domain_half_width=0.5,
        seed=seed_train,
    )

    objective.fit(X_train)
    y_train_tilde = objective.evaluate(X_train, standardized=True)

    solver = ConvexHullProjectionSolver(X_train)

    if config.sampling_strategy == "natural":
        warnings.warn(
            "Natural uniform sampling has been retired for open-loop extrapolation experiments. "
            "Use 'stratified' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        X_test, proj_res, _ = sample_natural_test(
            n_samples=config.n_test,
            dimension=config.dimension,
            solver=solver,
            seed=seed_test,
        )
        # Same distance bands as the stratified sampler.
        strata_labels = np.zeros(len(proj_res.d_norm), dtype=np.int64)
        strata_labels[(proj_res.d_norm >= 1e-6) & (proj_res.d_norm <= 0.15)] = 1
        strata_labels[(proj_res.d_norm > 0.15) & (proj_res.d_norm <= 0.40)] = 2
        strata_labels[proj_res.d_norm > 0.40] = 3
    else:
        n_per_stratum = config.n_test // 4
        X_test, proj_res, strata_labels = sample_stratified_test(
            n_per_stratum=n_per_stratum,
            dimension=config.dimension,
            X_train=X_train,
            solver=solver,
            seed=seed_test,
        )

    y_true = objective.evaluate(X_test, standardized=True)

    evaluator = MultiUQEvaluator(
        surrogate_type=config.surrogate_type,
        seed=seed_model,
        n_trees=config.n_trees,
        k=config.k,
        epsilon=config.eps,
        topological_decay_lambda=config.decay_lambda,
        eval_mode=config.eval_mode,
    )
    evaluator.fit(X_train, y_train_tilde)
    uq_res = evaluator.evaluate(X_test)

    abs_error = np.abs(y_true - uq_res.y_hat)
    n_pts = len(X_test)
    data_dict: dict[str, Any] = {
        "point_id": np.arange(n_pts, dtype=np.int64),
    }
    for d in range(config.dimension):
        data_dict[f"x_{d}"] = np.asarray(X_test[:, d], dtype=np.float64)
    data_dict.update(
        {
            "dimension": np.full(n_pts, config.dimension, dtype=np.int64),
            "n_train": np.full(n_pts, config.n_train, dtype=np.int64),
            "function_name": np.full(n_pts, objective.name, dtype=object),
            "seed": np.full(n_pts, config.seed, dtype=np.int64),
            "surrogate": config.surrogate_type,
            "stratum": strata_labels.astype(np.int64),
            "d_norm": np.asarray(proj_res.d_norm, dtype=np.float64),
            "d_rel": np.asarray(proj_res.d_rel, dtype=np.float64),
            "d_inf": np.asarray(proj_res.d_inf, dtype=np.float64),
            "is_interpolating": np.asarray(proj_res.is_interpolating, dtype=bool),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_hat": np.asarray(uq_res.y_hat, dtype=np.float64),
            "abs_error": np.asarray(abs_error, dtype=np.float64),
        }
    )

    for app in APPROACH_NAMES:
        data_dict[f"u_{app}"] = np.asarray(getattr(uq_res, f"u_{app}"), dtype=np.float64)
        data_dict[f"sigma_{app}"] = np.asarray(getattr(uq_res, f"sigma_{app}"), dtype=np.float64)
        data_dict[f"int_lwr_{app}"] = np.asarray(getattr(uq_res, f"int_lwr_{app}"), dtype=np.float64)
        data_dict[f"int_upr_{app}"] = np.asarray(getattr(uq_res, f"int_upr_{app}"), dtype=np.float64)
        for sfx in ("50", "80", "90", "95"):
            data_dict[f"int_lwr_{app}_{sfx}"] = np.asarray(
                getattr(uq_res, f"int_lwr_{app}_{sfx}"), dtype=np.float64
            )
            data_dict[f"int_upr_{app}_{sfx}"] = np.asarray(
                getattr(uq_res, f"int_upr_{app}_{sfx}"), dtype=np.float64
            )

    # Diagnostics and legacy column names.
    data_dict.update(
        {
            "u_hutter_between": np.asarray(uq_res.u_hutter_between, dtype=np.float64),
            "u_hutter_within": np.asarray(uq_res.u_hutter_within, dtype=np.float64),
            "u_shaker_total": np.asarray(uq_res.u_shaker_total, dtype=np.float64),
            "shaker_mi": np.asarray(uq_res.shaker_mi, dtype=np.float64),
            "shaker_total_entropy": np.asarray(uq_res.shaker_total_entropy, dtype=np.float64),
            "u_rf_fire_lower": np.asarray(uq_res.u_rf_fire_lower, dtype=np.float64),
            "u_slcb": np.asarray(uq_res.u_slcb, dtype=np.float64),
            "u_prox_a_lower": np.asarray(uq_res.u_prox_a_oob, dtype=np.float64),
            "u_prox_b_lower": np.asarray(uq_res.u_prox_b_oob, dtype=np.float64),
            "u_prox_ac_lower": np.asarray(uq_res.u_prox_ac_oob, dtype=np.float64),
            "u_prox_bc_lower": np.asarray(uq_res.u_prox_bc_oob, dtype=np.float64),
            "u_plcb_lower": np.asarray(uq_res.u_prox_a_oob, dtype=np.float64),
            "u_plcb": np.asarray(uq_res.u_prox_a_oob, dtype=np.float64),
            "delta_floor": np.asarray(uq_res.delta_floor, dtype=np.float64),
            "local_mae": np.asarray(uq_res.local_mae, dtype=np.float64),
            "q_lower": np.asarray(uq_res.q_lower, dtype=np.float64),
        }
    )
    point_df = pd.DataFrame(data_dict)

    if save_parquet and output_dir is not None:
        out_p = Path(output_dir)
        if out_p.suffix == ".parquet":
            out_p.parent.mkdir(parents=True, exist_ok=True)
            parquet_path = out_p
        else:
            out_p.mkdir(parents=True, exist_ok=True)
            filename = (
                f"extrapolation_{objective.name}"
                f"_d{config.dimension}"
                f"_n{config.n_train}"
                f"_{config.sampling_strategy}"
                f"_{config.surrogate_type}"
                f"_s{config.seed}.parquet"
            )
            parquet_path = out_p / filename
        point_df.to_parquet(parquet_path, index=False, engine="pyarrow")

    if compute_metrics:
        active_uncertainties = {
            # Every signal is scored under its short name and its ``u_`` name.
            "slcb": uq_res.u_slcb,
            "hutter_total": uq_res.u_hutter_total,
            "hutter_between": uq_res.u_hutter_between,
            "hutter_within": uq_res.u_hutter_within,
            "shaker_epistemic": uq_res.u_shaker_epistemic,
            "shaker_total": uq_res.u_shaker_total,
            "rf_fire": uq_res.u_rf_fire_lower,
            "prox_a": uq_res.u_prox_a_lower,
            "prox_b": uq_res.u_prox_b_lower,
            "prox_ac": uq_res.u_prox_ac_lower,
            "prox_bc": uq_res.u_prox_bc_lower,
            "plcb": uq_res.u_plcb_lower,
            "u_hutter_total": uq_res.u_hutter_total,
            "u_hutter_between": uq_res.u_hutter_between,
            "u_hutter_within": uq_res.u_hutter_within,
            "u_shaker_epistemic": uq_res.u_shaker_epistemic,
            "u_shaker_total": uq_res.u_shaker_total,
            "u_rf_fire_lower": uq_res.u_rf_fire_lower,
            "u_prox_a_lower": uq_res.u_prox_a_lower,
            "u_prox_b_lower": uq_res.u_prox_b_lower,
            "u_prox_ac_lower": uq_res.u_prox_ac_lower,
            "u_prox_bc_lower": uq_res.u_prox_bc_lower,
            "u_plcb_lower": uq_res.u_plcb_lower,
            "u_slcb": uq_res.u_slcb,
            "u_plcb": uq_res.u_plcb,
        }

        summary_dict = compute_comprehensive_metrics(
            y_true=y_true,
            y_hat=uq_res.y_hat,
            d_norm=proj_res.d_norm,
            strata_labels=strata_labels,
            uq_result=active_uncertainties,
        )
    else:
        summary_dict = {}

    summary_dict["dimension"] = config.dimension
    summary_dict["n_train"] = config.n_train
    summary_dict["function_name"] = config.function_name
    summary_dict["sampling_strategy"] = config.sampling_strategy
    summary_dict["seed"] = config.seed
    summary_dict["surrogate_type"] = config.surrogate_type
    summary_dict["surrogate"] = config.surrogate_type
    summary_dict["eval_mode"] = config.eval_mode
    summary_dict["n_test"] = len(point_df)

    return summary_dict, point_df
