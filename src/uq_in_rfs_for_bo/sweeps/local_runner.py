"""Run an extrapolation task file locally with a process pool (one single-threaded task per worker)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dyrf_bo.paths import REPO_ROOT
from dyrf_bo.sweeps.extrapolation_tasks import generate_tasks

DEFAULT_TASK_FILE = "results/extrapolation_sweep_tasks.txt"


def execute_command(task_id: int, cmd: str, dry_run: bool = False) -> dict[str, Any]:
    """Run one shell command; returns task_id, cmd, status ('dry_run'/'success'/'failed'), exit_code,
    elapsed_seconds and error."""
    result: dict[str, Any] = {"task_id": task_id, "cmd": cmd}
    if dry_run:
        return {**result, "status": "dry_run", "exit_code": 0, "elapsed_seconds": 0.0, "error": None}

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f".:{env.get('PYTHONPATH', '')}"

    start_time = time.perf_counter()
    try:
        proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {**result, "status": "failed", "exit_code": -1, "elapsed_seconds": elapsed, "error": str(e)}
    elapsed = time.perf_counter() - start_time
    if proc.returncode == 0:
        return {**result, "status": "success", "exit_code": 0, "elapsed_seconds": elapsed, "error": None}
    return {
        **result,
        "status": "failed",
        "exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "error": proc.stderr.strip() or proc.stdout.strip(),
    }


def _worker_task(args: tuple[int, str, bool]) -> dict[str, Any]:
    task_id, cmd, dry_run = args
    return execute_command(task_id=task_id, cmd=cmd, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extrapolation sweep tasks locally in parallel.")
    parser.add_argument(
        "--task-file",
        type=str,
        default=DEFAULT_TASK_FILE,
        help=f"Path to task file (default: {DEFAULT_TASK_FILE}).",
    )
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel worker processes (default: {default_workers}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Alias for --workers; takes precedence if both are given.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Generate and run the pilot suite (8 tasks).",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        default=False,
        help="Generate and run the 16-task stress suite.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to execute from the task file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log task commands without executing them.",
    )
    return parser


def run_local_sweep(
    task_file: str | Path | None = None,
    workers: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    pilot: bool = False,
    stress: bool = False,
) -> dict[str, Any]:
    """Run the tasks of ``task_file`` (or a freshly generated pilot/stress suite) in parallel.

    Lines starting with ``#`` are skipped. Returns counts, per-task results and the total runtime.
    """
    if pilot or stress:
        if task_file is None or str(task_file) == DEFAULT_TASK_FILE:
            task_path = Path("results/pilot_tasks.txt" if pilot else "results/stress_test_tasks.txt")
        else:
            task_path = Path(task_file)
        venv_py = REPO_ROOT / ".venv" / "bin" / "python"
        py_bin = str(venv_py) if venv_py.exists() else sys.executable

    if pilot:
        generate_tasks(output_file=task_path, pilot=True, python_bin=py_bin)
    elif stress:
        generate_tasks(
            output_file=task_path,
            stress=True,
            python_bin=py_bin,
            output_dir="results/extrapolation_uq/stress_raw",
            summary_dir="results/extrapolation_uq/stress_summaries",
        )
    else:
        task_path = Path(task_file if task_file is not None else DEFAULT_TASK_FILE)

    if not task_path.exists():
        raise FileNotFoundError(f"Task file '{task_path}' not found.")

    with open(task_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total_tasks = len(lines)
    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "executed_tasks": 0,
            "succeeded_tasks": 0,
            "failed_tasks": 0,
            "results": [],
            "elapsed_seconds": 0.0,
        }

    selected_lines = lines[:limit] if limit is not None else lines
    tasks_to_run = list(enumerate(selected_lines, start=1))
    n_exec = len(tasks_to_run)

    max_workers = workers if workers is not None else max(1, (os.cpu_count() or 1) - 1)
    print(
        f"Starting local sweep: {n_exec}/{total_tasks} tasks with {max_workers} worker processes (dry_run={dry_run})."
    )

    start_total = time.perf_counter()
    results: list[dict[str, Any]] = []

    task_inputs = [(task_id, cmd, dry_run) for task_id, cmd in tasks_to_run]
    with mp.Pool(processes=max_workers) as pool:
        completed = 0
        for res in pool.imap_unordered(_worker_task, task_inputs):
            results.append(res)
            completed += 1
            if res["status"] == "failed":
                print(f"[FAIL] Task {res['task_id']} exited with code {res['exit_code']}: {res['error']}")
            elif completed % 25 == 0 or completed == n_exec:
                print(f"Progress: [{completed}/{n_exec}] ({completed / n_exec * 100:.1f}%) completed.")

    elapsed_total = time.perf_counter() - start_total
    results.sort(key=lambda r: r["task_id"])

    succeeded = sum(1 for r in results if r["status"] in ("success", "dry_run"))
    failed = sum(1 for r in results if r["status"] == "failed")

    print(
        f"Completed sweep in {elapsed_total:.2f}s: "
        f"{succeeded} succeeded, {failed} failed out of {n_exec} executed."
    )

    return {
        "total_tasks": total_tasks,
        "executed_tasks": n_exec,
        "succeeded_tasks": succeeded,
        "failed_tasks": failed,
        "results": results,
        "elapsed_seconds": elapsed_total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workers = args.concurrency if args.concurrency is not None else args.workers

    summary = run_local_sweep(
        task_file=args.task_file,
        workers=workers,
        limit=args.limit,
        dry_run=args.dry_run,
        pilot=args.pilot,
        stress=args.stress,
    )

    return 0 if summary["failed_tasks"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
