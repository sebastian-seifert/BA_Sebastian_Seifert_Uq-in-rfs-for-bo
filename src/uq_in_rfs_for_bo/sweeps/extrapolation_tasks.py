"""Task manifest for the extrapolation UQ sweep (one ``scripts/run_extrapolation.py`` call per line).

Full grid: 8 dimensions x 6 training sizes x 4 functions x 30 seeds x 4 surrogates (23,040 tasks).
``--pilot`` and ``--stress`` write small fixed suites instead.
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import sys
from pathlib import Path

DEFAULT_DIMENSIONS = [2, 3, 5, 8, 16, 32, 64, 128]
DEFAULT_N_TRAINS = [34, 68, 136, 272, 544, 1088]
DEFAULT_FUNCTIONS = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DEFAULT_STRATEGIES = ["stratified"]
DEFAULT_SEEDS = list(range(30))
DEFAULT_SURROGATES = ["smac_default", "mature", "shallow", "coarse"]

PILOT_DIMENSIONS = [2, 16]
PILOT_N_TRAINS = [34]
PILOT_FUNCTIONS = ["sphere"]
PILOT_STRATEGIES = ["stratified"]
PILOT_SEEDS = [0]
PILOT_SURROGATES = ["smac_default", "mature", "shallow", "coarse"]

# Balanced 16-run design: each dimension sees every function once, two per training size.
STRESS_CONFIGURATIONS = [
    (2, 34, "sphere", "stratified", 0),
    (2, 34, "rosenbrock", "stratified", 0),
    (2, 68, "rastrigin", "stratified", 0),
    (2, 68, "ackley", "stratified", 0),
    (5, 34, "rosenbrock", "stratified", 0),
    (5, 34, "rastrigin", "stratified", 0),
    (5, 68, "ackley", "stratified", 0),
    (5, 68, "sphere", "stratified", 0),
    (16, 34, "rastrigin", "stratified", 0),
    (16, 34, "ackley", "stratified", 0),
    (16, 68, "sphere", "stratified", 0),
    (16, 68, "rosenbrock", "stratified", 0),
    (32, 34, "ackley", "stratified", 0),
    (32, 34, "sphere", "stratified", 0),
    (32, 68, "rosenbrock", "stratified", 0),
    (32, 68, "rastrigin", "stratified", 0),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the task list for the extrapolation UQ sweep.")
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/extrapolation_sweep_tasks.txt",
        help="Output task file (default: results/extrapolation_sweep_tasks.txt).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Pilot suite: D in [2, 16], N=34, sphere, seed 0, all surrogates.",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        default=False,
        help="16-run stress suite: D in [2, 5, 16, 32], all 4 functions, N in [34, 68].",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Passed as --output-dir to each task.")
    parser.add_argument("--summary-dir", type=str, default=None, help="Passed as --summary-dir to each task.")
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=DEFAULT_DIMENSIONS,
        help=f"Dimensions (default: {DEFAULT_DIMENSIONS}).",
    )
    parser.add_argument(
        "--n-trains",
        type=int,
        nargs="+",
        default=DEFAULT_N_TRAINS,
        help=f"Training sample sizes (default: {DEFAULT_N_TRAINS}).",
    )
    parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=DEFAULT_FUNCTIONS,
        help=f"Objective functions (default: {DEFAULT_FUNCTIONS}).",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help=f"Sampling strategies (default: {DEFAULT_STRATEGIES}).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds (default: 0..29)."
    )
    parser.add_argument(
        "--surrogates",
        type=str,
        nargs="+",
        choices=DEFAULT_SURROGATES,
        default=DEFAULT_SURROGATES,
        help=f"Surrogate model types (default: {DEFAULT_SURROGATES}).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python interpreter used in the task lines (default: 'python').",
    )
    parser.add_argument(
        "--skip-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append --skip-if-exists to each task (default: True).",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        default=False,
        help="Append --no-metrics to each task (write raw Parquet only).",
    )
    return parser


def generate_tasks(
    output_file: str | Path = "results/extrapolation_sweep_tasks.txt",
    pilot: bool = False,
    stress: bool = False,
    dimensions: list[int] | None = None,
    n_trains: list[int] | None = None,
    functions: list[str] | None = None,
    strategies: list[str] | None = None,
    seeds: list[int] | None = None,
    surrogates: list[str] | None = None,
    python_bin: str = "python",
    skip_if_exists: bool = True,
    output_dir: str | None = None,
    summary_dir: str | None = None,
    no_metrics: bool = False,
) -> list[str]:
    """Build the task lines and write them to ``output_file``.

    ``pilot`` and ``stress`` select the fixed suites and ignore the grid arguments (``pilot`` still
    honours ``surrogates``); otherwise ``None`` arguments fall back to the ``DEFAULT_*`` grid.
    """
    suffix = (
        (f" --output-dir {shlex.quote(output_dir)}" if output_dir else "")
        + (f" --summary-dir {shlex.quote(summary_dir)}" if summary_dir else "")
        + (" --skip-if-exists" if skip_if_exists else "")
        + (" --no-metrics" if no_metrics else "")
    )
    py_bin = shlex.quote(python_bin)

    def task_line(dim: int, n_train: int, func: str, strat: str, seed: int, surr: str) -> str:
        return (
            f"{py_bin} scripts/run_extrapolation.py "
            f"--dimension {dim} "
            f"--n-train {n_train} "
            f"--function {func} "
            f"--strategy {strat} "
            f"--seed {seed} "
            f"--surrogate {surr}"
            f"{suffix}"
        )

    if stress:
        tasks = [task_line(*cfg, "smac_default") for cfg in STRESS_CONFIGURATIONS]
    else:
        if pilot:
            grid = (
                PILOT_DIMENSIONS,
                PILOT_N_TRAINS,
                PILOT_FUNCTIONS,
                PILOT_STRATEGIES,
                PILOT_SEEDS,
                surrogates if surrogates is not None else PILOT_SURROGATES,
            )
        else:
            grid = (
                dimensions if dimensions is not None else DEFAULT_DIMENSIONS,
                n_trains if n_trains is not None else DEFAULT_N_TRAINS,
                functions if functions is not None else DEFAULT_FUNCTIONS,
                strategies if strategies is not None else DEFAULT_STRATEGIES,
                seeds if seeds is not None else DEFAULT_SEEDS,
                surrogates if surrogates is not None else DEFAULT_SURROGATES,
            )
        tasks = [task_line(*cfg) for cfg in itertools.product(*grid)]

    target_path = Path(output_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for cmd in tasks:
            f.write(cmd + "\n")

    print(f"Generated {len(tasks)} tasks in '{target_path}'.")
    return tasks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_tasks(
        output_file=args.output_file,
        pilot=args.pilot,
        stress=args.stress,
        dimensions=args.dimensions,
        n_trains=args.n_trains,
        functions=args.functions,
        strategies=args.strategies,
        seeds=args.seeds,
        surrogates=args.surrogates,
        python_bin=args.python_bin,
        skip_if_exists=args.skip_if_exists,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        no_metrics=args.no_metrics,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
