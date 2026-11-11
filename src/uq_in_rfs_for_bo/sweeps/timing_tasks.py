"""Task manifest for the BBOB timing sweep.

BBOB f8 (Rosenbrock), instance 0, d in {2, 4, 8, 16, 32}, all 11 optimizers x 20 seeds. The task order
is shuffled with a fixed seed so that runs of one dimension are not clustered in time on the cluster.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Sequence

from dyrf_bo.sweeps.bo_tasks import DEV_SUITE_OPTIMIZERS, build_task_line

TIMING_SWEEP_OUTPUT_DIR = "results/sweep_bbob_timing"
TIMING_SWEEP_RUNS_DIR = "runs/sweep_bbob_timing"
TIMING_DIMENSIONS = (2, 4, 8, 16, 32)
TIMING_FUNCTION_ID = 8
TIMING_INSTANCE_ID = 0


def generate_bbob_timing_sweep_tasks(
    seeds: int | Sequence[int] = 20,
    trials: int = 100,
    dimensions: Sequence[int] = TIMING_DIMENSIONS,
    function_id: int = TIMING_FUNCTION_ID,
    instance_id: int = TIMING_INSTANCE_ID,
    output_dir: str = TIMING_SWEEP_OUTPUT_DIR,
    runs_dir: str = TIMING_SWEEP_OUTPUT_DIR,
    baserundir: str = TIMING_SWEEP_RUNS_DIR,
    shuffle: bool = True,
    shuffle_seed: int = 42,
) -> list[str]:
    """Build the timing-sweep task lines and write them to ``<output_dir>/tasks.txt``."""
    seeds_list = list(range(1, seeds + 1)) if isinstance(seeds, int) else list(seeds)

    all_commands: list[str] = []
    for dim in dimensions:
        task_arg = f"+task=BBOB/cfg_{dim}_{function_id}_{instance_id}"
        for opt_info in DEV_SUITE_OPTIMIZERS:
            for seed in seeds_list:
                all_commands.append(
                    build_task_line(
                        opt_info, task_arg, seed, trials, telemetry_dir=runs_dir, baserundir=baserundir
                    )
                )

    if shuffle:
        random.Random(shuffle_seed).shuffle(all_commands)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "tasks.txt"), "w", encoding="utf-8") as f:
            for cmd in all_commands:
                f.write(f"{cmd}\n")

    return all_commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the BBOB timing-sweep task manifest.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per run (default: 100)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TIMING_SWEEP_OUTPUT_DIR,
        help=f"Target output directory (default: {TIMING_SWEEP_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baserundir",
        type=str,
        default=TIMING_SWEEP_RUNS_DIR,
        help=f"Base run output directory (default: {TIMING_SWEEP_RUNS_DIR})",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42, help="Seed for shuffling the task order (default: 42)"
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Keep the tasks in generation order")
    args = parser.parse_args()

    tasks = generate_bbob_timing_sweep_tasks(
        seeds=args.seeds,
        trials=args.trials,
        output_dir=args.output_dir,
        runs_dir=args.output_dir,
        baserundir=args.baserundir,
        shuffle=not args.no_shuffle,
        shuffle_seed=args.shuffle_seed,
    )
    print(f"Generated {len(tasks)} timing sweep tasks in '{args.output_dir}/tasks.txt'")


if __name__ == "__main__":
    main()
