"""Run ``scripts/run_bo.py`` and record wall-clock time, BO overhead and hardware info.

The result is written to ``timing_<name>.json`` next to the run's telemetry path.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dyrf_bo.paths import REPO_ROOT


def collect_hardware_metadata() -> dict[str, Any]:
    """Host, CPU, memory, platform and thread-count settings."""
    cpu_model = "Unknown"
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            with open(cpuinfo_path, encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    if cpu_model == "Unknown":
        cpu_model = platform.processor() or platform.machine()

    total_ram_gb = None
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            with open(meminfo_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(re.search(r"\d+", line).group())
                        total_ram_gb = round(kb / (1024 * 1024), 2)
                        break
        except Exception:
            pass

    affinity_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except Exception:
            pass

    thread_env_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    thread_env = {v: os.environ.get(v) for v in thread_env_vars if v in os.environ}

    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": affinity_count,
        "total_ram_gb": total_ram_gb,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "thread_env": thread_env,
    }


def compute_timing_metrics_from_trial_logs(
    trial_logs_path: str,
    total_wallclock_seconds: float,
) -> dict[str, Any]:
    """Sum the objective evaluation times in trial_logs.jsonl; the rest of the wall-clock time is BO overhead."""
    target_eval_seconds = 0.0
    n_trials = 0
    p = Path(trial_logs_path)

    if p.is_file():
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        tval = record.get("trial_value", {})
                        if isinstance(tval, dict):
                            t = tval.get("time")
                            if t is not None:
                                target_eval_seconds += float(t)
                        n_trials += 1
                    except Exception:
                        continue
        except Exception:
            pass

    bo_overhead_seconds = max(0.0, total_wallclock_seconds - target_eval_seconds)

    return {
        "n_trials_logged": n_trials,
        "target_eval_seconds": round(target_eval_seconds, 6),
        "bo_overhead_seconds": round(bo_overhead_seconds, 6),
    }


def parse_task_and_telemetry_info(args: Sequence[str]) -> dict[str, Any]:
    """Optimizer, task (with BBOB dimension/function/instance), seed and output paths from Hydra arguments."""
    info: dict[str, Any] = {
        "optimizer_id": None,
        "task": None,
        "dimension": None,
        "function_id": None,
        "instance_id": None,
        "seed": None,
        "telemetry_path": None,
        "timing_path": None,
        "baserundir": None,
    }

    for arg in args:
        if arg.startswith("optimizer_id="):
            info["optimizer_id"] = arg.split("=", 1)[1]
        elif arg.startswith("seed="):
            try:
                info["seed"] = int(arg.split("=", 1)[1])
            except ValueError:
                pass
        elif arg.startswith("+task="):
            task_str = arg.split("=", 1)[1]
            info["task"] = task_str
            m = re.search(r"cfg_(\d+)_(\d+)_(\d+)", task_str)
            if m:
                info["dimension"] = int(m.group(1))
                info["function_id"] = int(m.group(2))
                info["instance_id"] = int(m.group(3))
        elif arg.startswith("++optimizer.telemetry_path="):
            t_path = arg.split("=", 1)[1]
            info["telemetry_path"] = t_path
            parent = os.path.dirname(t_path)
            base = os.path.basename(t_path)
            if base.startswith("telemetry_"):
                timing_name = base.replace("telemetry_", "timing_", 1)
            else:
                timing_name = f"timing_{base}"
            info["timing_path"] = os.path.join(parent, timing_name)
        elif arg.startswith("baserundir="):
            info["baserundir"] = arg.split("=", 1)[1]

    return info


def execute_timed_run(args: Sequence[str]) -> dict[str, Any]:
    """Run ``scripts/run_bo.py`` with ``args``, time it and write the timing JSON."""
    parsed_info = parse_task_and_telemetry_info(args)
    target_timing_path = parsed_info.get("timing_path")

    hardware = collect_hardware_metadata()

    t_start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_bo.py")] + list(args))
    exit_code = proc.returncode
    t_end = time.perf_counter()
    wallclock_seconds = round(t_end - t_start, 6)

    trial_logs_candidates = []
    if parsed_info.get("baserundir"):
        bdir = Path(parsed_info["baserundir"])
        if bdir.exists():
            trial_logs_candidates.extend(bdir.glob("**/trial_logs.jsonl"))

    trial_metrics = {
        "n_trials_logged": 0,
        "target_eval_seconds": 0.0,
        "bo_overhead_seconds": wallclock_seconds,
    }
    if trial_logs_candidates:
        latest_log = max(trial_logs_candidates, key=os.path.getmtime)
        trial_metrics = compute_timing_metrics_from_trial_logs(str(latest_log), wallclock_seconds)

    result_payload: dict[str, Any] = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "wallclock_seconds": wallclock_seconds,
        "bo_overhead_seconds": trial_metrics["bo_overhead_seconds"],
        "target_eval_seconds": trial_metrics["target_eval_seconds"],
        "n_trials_logged": trial_metrics["n_trials_logged"],
        "task_info": parsed_info,
        "hardware": hardware,
        "args": list(args),
    }

    if target_timing_path:
        p = Path(target_timing_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

    return result_payload


def main() -> None:
    args = sys.argv[1:]
    result = execute_timed_run(args)
    sys.exit(result["exit_code"])


if __name__ == "__main__":
    main()
