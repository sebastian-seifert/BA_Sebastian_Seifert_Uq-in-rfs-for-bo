"""Command-line interface for one extrapolation UQ experiment.

Writes the per-point parquet table and the JSON scorecard for one
(dimension, n_train, function, strategy, surrogate, seed) configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dyrf_bo.extrapolation.runner import (
    ExtrapolationRunConfig,
    run_single_experiment,
)


def _json_serializable(val: Any) -> Any:
    """Recursively convert NumPy scalars and collections to standard Python types."""
    if isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [_json_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_serializable(x) for x in val]
    return val


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for running a single extrapolation experiment."""
    parser = argparse.ArgumentParser(
        description="Run a single extrapolation uncertainty quantification experiment."
    )
    # Required arguments
    parser.add_argument(
        "--dimension",
        type=int,
        required=True,
        help="Feature space dimensionality D (e.g. 2, 3, 5, 8, 16, 32).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        required=True,
        help="Training sample size N in [-0.5, 0.5]^D (e.g. 34, 68, 136, 272, 544, 1088).",
    )
    parser.add_argument(
        "--function",
        type=str.lower,
        required=True,
        choices=["sphere", "rosenbrock", "rastrigin", "ackley"],
        help="Benchmark objective function name ('sphere', 'rosenbrock', 'rastrigin', 'ackley').",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="stratified",
        choices=["stratified", "natural"],
        help="Test sampling strategy ('stratified' [default]; 'natural' retired for open-loop extrapolation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for sampling and surrogate initialization.",
    )

    # Optional / hyperparameter arguments
    parser.add_argument(
        "--n-test",
        type=int,
        default=10000,
        help="Total test sample size (default: 10000).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=28,
        help="Proximity nearest-neighbor count k (default: 28).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.080791,
        help="Proximity exploration floor coefficient epsilon (default: 0.080791).",
    )
    parser.add_argument(
        "--decay-lambda",
        type=float,
        default=0.20486,
        help="Topological decay parameter lambda (default: 0.20486).",
    )
    parser.add_argument(
        "--n-trees",
        type=int,
        default=10,
        help="Number of trees in ensemble surrogate (default: 10).",
    )
    parser.add_argument(
        "--surrogate",
        "--surrogate-type",
        dest="surrogate",
        type=str.lower,
        default="smac_default",
        choices=["smac_default", "mature", "shallow", "coarse"],
        help="Surrogate model type (default: 'smac_default').",
    )

    # Output paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/extrapolation_uq/raw",
        help="Directory to save raw Parquet evaluations (default: results/extrapolation_uq/raw).",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="results/extrapolation_uq/summaries",
        help="Directory to save summary JSON scorecard (default: results/extrapolation_uq/summaries).",
    )

    # Flags
    parser.add_argument(
        "--save-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw evaluations as Parquet (default: True).",
    )
    parser.add_argument(
        "--no-parquet",
        dest="save_parquet",
        action="store_false",
        help="Alias to disable saving raw Parquet evaluations.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        default=False,
        help="Skip execution if target output files already exist (default: False).",
    )
    parser.add_argument(
        "--eval-mode",
        type=str.lower,
        default="all",
        choices=["all", "unweighted_proximity_only"],
        help="UQ evaluation mode: 'all' or 'unweighted_proximity_only' (default: 'all').",
    )
    parser.add_argument(
        "--unweighted-proximity-only",
        action="store_true",
        default=False,
        help="Fast mode: compute only unweighted proximity UQ, skipping Shaker numerical integration.",
    )
    parser.add_argument(
        "--compute-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute calibration and ranking metrics (default: True).",
    )
    parser.add_argument(
        "--no-metrics",
        dest="compute_metrics",
        action="store_false",
        help="Disable metric computation and skip saving JSON summary.",
    )
    parser.add_argument(
        "--save-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save summary scorecard to JSON (default: True).",
    )
    parser.add_argument(
        "--no-summary",
        dest="save_summary",
        action="store_false",
        help="Disable saving summary scorecard JSON.",
    )

    return parser


def run_experiment(args: argparse.Namespace) -> int:
    """Execute experiment according to parsed arguments."""
    func = args.function.lower()
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)

    parquet_filename = (
        f"extrapolation_{func}"
        f"_d{args.dimension}"
        f"_n{args.n_train}"
        f"_{args.strategy}"
        f"_{args.surrogate}"
        f"_s{args.seed}.parquet"
    )
    summary_filename = (
        f"summary_{func}_d{args.dimension}_n{args.n_train}_{args.strategy}_{args.surrogate}_s{args.seed}.json"
    )

    parquet_path = output_dir / parquet_filename
    summary_path = summary_dir / summary_filename

    should_save_summary = args.save_summary and args.compute_metrics

    if args.skip_if_exists:
        if args.save_parquet and should_save_summary:
            both_exist = (
                parquet_path.is_file()
                and parquet_path.stat().st_size > 0
                and summary_path.is_file()
                and summary_path.stat().st_size > 0
            )
        elif args.save_parquet:
            both_exist = parquet_path.is_file() and parquet_path.stat().st_size > 0
        elif should_save_summary:
            both_exist = summary_path.is_file() and summary_path.stat().st_size > 0
        else:
            both_exist = False

        if both_exist:
            print(
                f"[SKIP] Experiment {func} (d={args.dimension}, n={args.n_train}, "
                f"{args.strategy}, {args.surrogate}, s={args.seed}) target outputs already exist."
            )
            if args.save_parquet:
                print(f"       Parquet: {parquet_path}")
            if should_save_summary:
                print(f"       Summary: {summary_path}")
            return 0

    eval_mode = "unweighted_proximity_only" if args.unweighted_proximity_only else args.eval_mode
    config = ExtrapolationRunConfig(
        dimension=args.dimension,
        n_train=args.n_train,
        function_name=func,
        sampling_strategy=args.strategy,
        seed=args.seed,
        n_test=args.n_test,
        k=args.k,
        eps=args.eps,
        decay_lambda=args.decay_lambda,
        n_trees=args.n_trees,
        surrogate_type=args.surrogate,
        eval_mode=eval_mode,
    )

    start_time = time.perf_counter()
    summary_dict, _ = run_single_experiment(
        config=config,
        output_dir=output_dir,
        save_parquet=args.save_parquet,
        compute_metrics=args.compute_metrics,
    )
    elapsed_time = time.perf_counter() - start_time
    summary_dict["elapsed_seconds"] = elapsed_time

    if should_save_summary:
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_json_serializable(summary_dict), f, indent=2)

    slcb_picp = summary_dict.get("slcb_picp", float("nan"))
    plcb_picp = summary_dict.get("plcb_picp", float("nan"))
    status_suffix = f" | SLCB PICP={slcb_picp:.4f}, PLCB PICP={plcb_picp:.4f}" if args.compute_metrics else ""
    print(
        f"[DONE] Completed {func} (d={args.dimension}, n={args.n_train}, "
        f"{args.strategy}, {args.surrogate}, s={args.seed}) in {elapsed_time:.2f}s"
        f"{status_suffix}"
    )
    if args.save_parquet:
        print(f"       Parquet: {parquet_path}")
    if should_save_summary:
        print(f"       Summary: {summary_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
