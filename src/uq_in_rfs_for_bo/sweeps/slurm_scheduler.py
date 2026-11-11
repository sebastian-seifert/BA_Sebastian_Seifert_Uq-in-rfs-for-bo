"""Submit a task manifest to Slurm as array-job chunks without exceeding the queue limits.

The queue is polled with ``squeue --me``; new chunks are submitted once it drains below a trigger
threshold. Progress is stored in a JSON state file so the scheduler can be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dyrf_bo.sweeps.bo_tasks import prepare_super_sweep


def _count_bracketed_array_tasks(bracket_spec: str) -> int:
    """Number of tasks in a Slurm array spec such as '1-200%25' or '1-10,15,20-25:2'."""
    inner = bracket_spec.strip().lstrip("[").rstrip("]")
    if not inner:
        return 1
    inner = inner.split("%")[0]
    total = 0
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            if ":" in part:
                range_part, step_str = part.split(":", 1)
                step = max(1, int(step_str))
            else:
                range_part = part
                step = 1
            start_str, end_str = range_part.split("-", 1)
            start_idx = int(start_str)
            end_idx = int(end_str)
            if end_idx >= start_idx:
                total += len(range(start_idx, end_idx + 1, step))
        else:
            total += 1
    return max(1, total)


def parse_squeue_me_output(
    output: str,
    ignore_job_ids: Sequence[str] | None = None,
) -> tuple[int, int]:
    """Count (distinct jobs, array tasks) in the output of ``squeue --me -h -o '%i %t'``."""
    lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
    if not lines:
        return 0, 0

    ignore_set = {str(jid).split("_")[0].split("[")[0] for jid in (ignore_job_ids or []) if jid}

    job_ids = set()
    total_tasks = 0

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        full_id = parts[0]
        base_id = full_id.split("_")[0].split("[")[0]
        if base_id in ignore_set:
            continue
        job_ids.add(base_id)
        if "[" in full_id and "]" in full_id:
            bracket_part = full_id[full_id.index("[") + 1 : full_id.rindex("]")]
            total_tasks += _count_bracketed_array_tasks(bracket_part)
        else:
            total_tasks += 1

    return len(job_ids), total_tasks


@dataclass
class SchedulerConfig:
    task_manifest: str = "results/sweep_bbob_full_comparison/tasks_all.txt"
    sbatch_script: str = ""
    chunk_size: int = 250
    max_active_tasks: int = 4500
    min_remaining_jobs_trigger: int = 10
    min_active_tasks_trigger: int | None = None
    max_batch_tasks: int | None = None
    use_task_offset: bool = False
    array_concurrency: int = 25
    poll_interval_sec: int = 60
    state_file: str = "results/sweep_bbob_full_comparison/scheduler_state.json"
    job_name_prefix: str = "bbob_swp"


class MasterSlurmScheduler:
    """Chunked array-job submission driven by the current queue occupancy."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config

    def calculate_chunks(self, total_tasks: int) -> list[tuple[int, int]]:
        """1-based inclusive (start, end) ranges of at most ``chunk_size`` tasks."""
        chunks: list[tuple[int, int]] = []
        chunk_size = self.config.chunk_size
        for start in range(1, total_tasks + 1, chunk_size):
            end = min(start + chunk_size - 1, total_tasks)
            chunks.append((start, end))
        return chunks

    def load_state(self) -> dict[str, Any]:
        """State from ``state_file``, or a fresh state if it is missing or unreadable."""
        if os.path.exists(self.config.state_file):
            try:
                with open(self.config.state_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_or_submitted_chunks": 0,
            "submitted_chunks_info": [],
            "all_chunks_dispatched": False,
        }

    def save_state(self, state: dict[str, Any]) -> None:
        state_dir = os.path.dirname(self.config.state_file)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        with open(self.config.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def query_squeue_me(self) -> tuple[int, int]:
        """(jobs, tasks) currently queued for this user, excluding the scheduler's own job."""
        try:
            res = subprocess.run(
                ["squeue", "--me", "-h", "-o", "%i %t"],
                capture_output=True,
                text=True,
                check=True,
            )
            own_job_id = os.environ.get("SLURM_JOB_ID")
            ignore_ids = [own_job_id] if own_job_id else None
            return parse_squeue_me_output(res.stdout, ignore_job_ids=ignore_ids)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0, 0

    def submit_chunk(
        self,
        chunk_idx: int,
        start: int,
        end: int,
        dry_run: bool = False,
    ) -> str | None:
        """Submit one chunk with sbatch and return its job ID (None on failure)."""
        job_name = f"{self.config.job_name_prefix}_c{chunk_idx}"
        if self.config.use_task_offset:
            task_offset = start - 1
            chunk_len = end - start + 1
            export_arg = f"--export=ALL,TASK_FILE={self.config.task_manifest},TASK_OFFSET={task_offset}"
            array_arg = f"--array=1-{chunk_len}%{self.config.array_concurrency}"
        else:
            export_arg = f"--export=ALL,TASK_FILE={self.config.task_manifest}"
            array_arg = f"--array={start}-{end}%{self.config.array_concurrency}"

        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            export_arg,
            array_arg,
            self.config.sbatch_script,
        ]
        if dry_run:
            print(f"[DRY-RUN] Executing: {' '.join(cmd)}")
            return f"mock_job_{chunk_idx}"

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = res.stdout.strip()
            return job_id
        except Exception as e:
            print(f"[ERROR] Failed to submit chunk {chunk_idx} ({start}-{end}): {e}", file=sys.stderr)
            return None

    def step(self, dry_run: bool = False) -> list[tuple[int, int, int]]:
        """Run one scheduling decision; returns the (chunk_idx, start, end) chunks submitted."""
        if not os.path.exists(self.config.task_manifest) and (
            "sweep_bo_uq_alignment_super" in self.config.task_manifest
            or "sweep_bbsubset_dev_big_comparison_fix" in self.config.task_manifest
        ):
            print(
                f"[INFO] Manifest '{self.config.task_manifest}' not found. Auto-generating BO UQ super-sweep tasks..."
            )
            results_root = os.path.dirname(os.path.dirname(self.config.task_manifest)) or "results"
            prepare_super_sweep(output_results_dir=results_root)

        if not os.path.exists(self.config.task_manifest):
            raise FileNotFoundError(f"Manifest not found: {self.config.task_manifest}")

        with open(self.config.task_manifest, encoding="utf-8") as f:
            total_tasks = sum(1 for line in f if line.strip())

        chunks = self.calculate_chunks(total_tasks)
        total_chunk_count = len(chunks)

        state = self.load_state()
        next_chunk_idx = state.get("completed_or_submitted_chunks", 0)

        if next_chunk_idx >= total_chunk_count:
            state["all_chunks_dispatched"] = True
            self.save_state(state)
            return []

        remaining_jobs, active_tasks = self.query_squeue_me()

        print(f"[STATUS] Cluster queue: {remaining_jobs} jobs, {active_tasks} tasks currently active.")

        submitted_in_step: list[tuple[int, int, int]] = []

        # Trigger on the active-task count if a task threshold is given, otherwise on the job count.
        if self.config.min_active_tasks_trigger is not None:
            should_trigger = active_tasks < self.config.min_active_tasks_trigger
        else:
            should_trigger = remaining_jobs < self.config.min_remaining_jobs_trigger

        if should_trigger:
            available_task_capacity = max(0, self.config.max_active_tasks - active_tasks)
            if self.config.max_batch_tasks is not None:
                # With a task trigger and max_active_tasks <= max_batch_tasks, the batch size alone
                # limits the submission (the queue count is not subtracted).
                if (
                    self.config.min_active_tasks_trigger is not None
                    and self.config.max_active_tasks <= self.config.max_batch_tasks
                ):
                    available_task_capacity = self.config.max_batch_tasks
                else:
                    available_task_capacity = min(available_task_capacity, self.config.max_batch_tasks)

            if self.config.min_active_tasks_trigger is not None:
                print(
                    f"[TRIGGER] Active tasks ({active_tasks}) < {self.config.min_active_tasks_trigger}. "
                    f"Available task capacity: {available_task_capacity}."
                )
            else:
                print(
                    f"[TRIGGER] Remaining jobs ({remaining_jobs}) < {self.config.min_remaining_jobs_trigger}. "
                    f"Available task capacity: {available_task_capacity}."
                )

            while next_chunk_idx < total_chunk_count:
                start, end = chunks[next_chunk_idx]
                chunk_len = end - start + 1

                if chunk_len > available_task_capacity:
                    print(
                        f"[CAP] Next chunk {next_chunk_idx} requires {chunk_len} tasks, but capacity is {available_task_capacity}. Pausing submissions."
                    )
                    break

                job_id = self.submit_chunk(next_chunk_idx, start, end, dry_run=dry_run)
                if job_id is not None:
                    submitted_in_step.append((next_chunk_idx, start, end))
                    state["submitted_chunks_info"].append(
                        {
                            "chunk_idx": next_chunk_idx,
                            "start": start,
                            "end": end,
                            "job_id": job_id,
                        }
                    )
                    next_chunk_idx += 1
                    available_task_capacity -= chunk_len
                    active_tasks += chunk_len
                    state["completed_or_submitted_chunks"] = next_chunk_idx
                    self.save_state(state)
                else:
                    break

            state["completed_or_submitted_chunks"] = next_chunk_idx
            if next_chunk_idx >= total_chunk_count:
                state["all_chunks_dispatched"] = True
            self.save_state(state)

        return submitted_in_step

    def run_loop(self, dry_run: bool = False, single_step: bool = False) -> None:
        """Poll every ``poll_interval_sec`` until all chunks are dispatched (or once if ``single_step``)."""
        print(f"Starting Master Slurm Scheduler (Manifest: {self.config.task_manifest})...")
        while True:
            submitted = self.step(dry_run=dry_run)
            state = self.load_state()
            if submitted:
                print(
                    f"[DISPATCH] Submitted {len(submitted)} chunks. Total dispatched: "
                    f"{state['completed_or_submitted_chunks']} chunks."
                )
            if state.get("all_chunks_dispatched", False):
                print("[DONE] All task chunks have been dispatched to Slurm!")
                break
            if single_step:
                break
            time.sleep(self.config.poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a task manifest to Slurm in array chunks.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sbatch", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-tasks", type=int, default=4500)
    parser.add_argument("--batch-tasks", type=int, default=None)
    parser.add_argument("--trigger-jobs", type=int, default=10)
    parser.add_argument("--trigger-tasks", type=int, default=None)
    parser.add_argument("--use-task-offset", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--job-prefix", "--job-name-prefix", dest="job_prefix", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")

    args = parser.parse_args()

    default_state_file = (
        args.state_file
        if args.state_file
        else os.path.join(os.path.dirname(args.manifest) or ".", "scheduler_state.json")
    )
    if args.job_prefix:
        default_job_prefix = args.job_prefix
    else:
        dim_match = re.search(r"dim(\d+)", args.manifest)
        default_job_prefix = f"bbob_d{dim_match.group(1)}" if dim_match else "bbob_swp"

    cfg = SchedulerConfig(
        task_manifest=args.manifest,
        sbatch_script=args.sbatch,
        chunk_size=args.chunk_size,
        max_active_tasks=args.max_tasks,
        min_remaining_jobs_trigger=args.trigger_jobs,
        min_active_tasks_trigger=args.trigger_tasks,
        max_batch_tasks=args.batch_tasks,
        use_task_offset=args.use_task_offset,
        array_concurrency=args.concurrency,
        poll_interval_sec=args.interval,
        state_file=default_state_file,
        job_name_prefix=default_job_prefix,
    )
    scheduler = MasterSlurmScheduler(cfg)
    scheduler.run_loop(dry_run=args.dry_run, single_step=args.once)


if __name__ == "__main__":
    main()
