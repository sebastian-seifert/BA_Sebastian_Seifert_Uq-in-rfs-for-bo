"""Task manifests for the BBSubset-dev sweep and the BO super-sweep (10 further suites).

Each suite is written to ``<results>/<sweep>_fix/tasks_all.txt``; all suites except BBSubset-dev are
also concatenated into ``<results>/sweep_bo_uq_alignment_super/tasks_all.txt``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from dyrf_bo.benchmarks.bbob import CarpsBBOBFullRegistry
from dyrf_bo.benchmarks.bbsubset import CarpsBBSubsetRegistry
from dyrf_bo.benchmarks.realworld import CarpsRealworldRegistry


def _optimizer(optimizer_id: str, config: str) -> dict[str, str]:
    return {"id": optimizer_id, "container_id": optimizer_id, "config_arg": f"+optimizer={config}"}


BASELINE_OPTIMIZERS: list[dict[str, str]] = [
    _optimizer("SMAC3_HPOFacade_lcb", "smac3_hpo_facade_lcb"),
    _optimizer("SMAC3_HPOFacade_ei", "smac3_hpo_facade_ei"),
    _optimizer("SMAC20_Entropy_LCB", "smac20_entropy_lcb"),
]

# OOB variants run on every suite; the CV variants only on BBSubset-dev.
OOB_PROXIMITY_OPTIMIZERS: list[dict[str, str]] = [
    _optimizer("SMAC20_ProximityA_LCB", "smac20_proximity_a_lcb"),
    _optimizer("SMAC20_ProximityAC_LCB", "smac20_proximity_ac_lcb"),
    _optimizer("SMAC20_ProximityB_LCB", "smac20_proximity_b_lcb"),
    _optimizer("SMAC20_ProximityBC_LCB", "smac20_proximity_bc_lcb"),
]

DEV_PROXIMITY_OPTIMIZERS: list[dict[str, str]] = [
    _optimizer("SMAC20_ProximityA_LCB", "smac20_proximity_a_lcb"),
    _optimizer("SMAC20_ProximityA_LCB_CV", "smac20_proximity_a_lcb_cv"),
    _optimizer("SMAC20_ProximityAC_LCB", "smac20_proximity_ac_lcb"),
    _optimizer("SMAC20_ProximityAC_LCB_CV", "smac20_proximity_ac_lcb_cv"),
    _optimizer("SMAC20_ProximityB_LCB", "smac20_proximity_b_lcb"),
    _optimizer("SMAC20_ProximityB_LCB_CV", "smac20_proximity_b_lcb_cv"),
    _optimizer("SMAC20_ProximityBC_LCB", "smac20_proximity_bc_lcb"),
    _optimizer("SMAC20_ProximityBC_LCB_CV", "smac20_proximity_bc_lcb_cv"),
]

OOB_SUITE_OPTIMIZERS: list[dict[str, str]] = BASELINE_OPTIMIZERS + OOB_PROXIMITY_OPTIMIZERS
DEV_SUITE_OPTIMIZERS: list[dict[str, str]] = BASELINE_OPTIMIZERS + DEV_PROXIMITY_OPTIMIZERS

DEV_SUITE = "sweep_bbsubset_dev_big_comparison"


def _suite(
    sweep_name: str,
    optimizers: list[dict[str, str]],
    tasks_fn: Callable[[], list[str]],
    expected_count: int,
) -> dict[str, Any]:
    return {
        "sweep_name": sweep_name,
        "optimizers": optimizers,
        "tasks_fn": tasks_fn,
        "seeds": 20,
        "trials": 100,
        "expected_count": expected_count,
    }


_bbob_dim = CarpsBBOBFullRegistry.get_tasks_by_dimension

# Listed in submission priority order.
EXPECTED_SUITE_SPECS: list[dict[str, Any]] = [
    _suite(DEV_SUITE, DEV_SUITE_OPTIMIZERS, CarpsBBSubsetRegistry.get_dev_tasks, 4400),
    _suite("sweep_bbsubset_test", OOB_SUITE_OPTIMIZERS, CarpsBBSubsetRegistry.get_test_tasks, 2800),
    _suite("sweep_yahpo_rbv2_ranger", OOB_SUITE_OPTIMIZERS, CarpsRealworldRegistry.get_ranger_tasks, 16660),
    _suite("sweep_bbob_dim8", OOB_SUITE_OPTIMIZERS, partial(_bbob_dim, 8), 10080),
    _suite("sweep_bbob_dim16", OOB_SUITE_OPTIMIZERS, partial(_bbob_dim, 16), 10080),
    _suite("sweep_bbob_dim32", OOB_SUITE_OPTIMIZERS, partial(_bbob_dim, 32), 10080),
    _suite("sweep_bbob_dim2", OOB_SUITE_OPTIMIZERS, partial(_bbob_dim, 2), 10080),
    _suite("sweep_bbob_dim4", OOB_SUITE_OPTIMIZERS, partial(_bbob_dim, 4), 10080),
    _suite("sweep_bbob_dim128", OOB_SUITE_OPTIMIZERS, partial(_bbob_dim, 128), 10080),
    _suite("sweep_yahpo_rbv2_super", OOB_SUITE_OPTIMIZERS, CarpsRealworldRegistry.get_super_tasks, 14420),
    _suite("sweep_hpobench_ml", OOB_SUITE_OPTIMIZERS, CarpsRealworldRegistry.get_hpobench_ml_tasks, 12320),
]

SUITE_NAMES: list[str] = [spec["sweep_name"] for spec in EXPECTED_SUITE_SPECS]

FIVE_METHODS: list[str] = [
    "SMAC20_ProximityAC_LCB",
    "SMAC20_ProximityB_LCB",
    "SMAC3_HPOFacade_ei",
    "SMAC3_HPOFacade_lcb",
    "SMAC20_Entropy_LCB",
]


def build_task_line(
    opt_info: dict[str, str],
    task_arg: str,
    seed: int,
    trials: int,
    telemetry_dir: str,
    baserundir: str,
) -> str:
    """Hydra arguments for one ``scripts/run_bo.py`` call."""
    task_name = task_arg.split("/")[-1]
    task_cmd = f"+{task_arg}" if not task_arg.startswith("+") else task_arg
    opt_id = opt_info["id"]
    telemetry_path = f"{telemetry_dir}/telemetry_{opt_id}_{task_name}_seed{seed}.json"
    return (
        f"--config-dir configs "
        f"{opt_info['config_arg']} "
        f"{task_cmd} "
        f"task.optimization_resources.n_trials={trials} "
        f"seed={seed} "
        f"++optimizer.telemetry_path={telemetry_path} "
        f"baserundir={baserundir} "
        f"optimizer_id={opt_id} "
        f"optimizer_container_id={opt_info['container_id']}"
    )


def rewrite_line_to_fix(line: str, sweep_name: str) -> str:
    """Point the telemetry and baserundir paths of an existing task line to ``<sweep>_fix``."""
    rewritten = re.sub(
        rf"(?<=telemetry_path=)results/{re.escape(sweep_name)}(?=/telemetry_)",
        f"results/{sweep_name}_fix",
        line.strip(),
    )
    return re.sub(
        rf"(?<=baserundir=)runs/{re.escape(sweep_name)}\b",
        f"runs/{sweep_name}_fix",
        rewritten,
    )


def prepare_suite_tasks(spec: dict[str, Any], source_results_dir: Path = Path("results")) -> list[str]:
    """Task lines of one suite.

    Matching lines of an existing ``<source>/<sweep>/tasks_all.txt`` (or ``tasks.txt``) are kept first
    (rewritten to ``_fix`` paths); missing optimizer/task/seed combinations are appended in a fixed order.
    """
    sweep_name: str = spec["sweep_name"]
    optimizers: Sequence[dict[str, str]] = spec["optimizers"]
    n_seeds: int = spec["seeds"]
    trials: int = spec["trials"]
    expected_count: int = spec["expected_count"]

    allowed_config_args = {opt["config_arg"] for opt in optimizers}
    allowed_seed_tokens = {f"seed={s}" for s in range(1, n_seeds + 1)}

    source_suite_dir = source_results_dir / sweep_name
    filtered_lines: list[str] = []
    for candidate in (source_suite_dir / "tasks_all.txt", source_suite_dir / "tasks.txt"):
        if candidate.is_file():
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                tokens = stripped.split()
                if any(tok in allowed_config_args for tok in tokens) and any(
                    tok in allowed_seed_tokens for tok in tokens
                ):
                    filtered_lines.append(rewrite_line_to_fix(stripped, sweep_name))
            break

    if len(filtered_lines) == expected_count and len(set(filtered_lines)) == expected_count:
        return filtered_lines

    existing_set = set(filtered_lines)
    tasks_list = spec["tasks_fn"]()
    result_lines = list(filtered_lines)
    for opt_info in optimizers:
        for task_arg in tasks_list:
            for seed in range(1, n_seeds + 1):
                cmd = build_task_line(
                    opt_info,
                    task_arg,
                    seed,
                    trials,
                    telemetry_dir=f"results/{sweep_name}_fix",
                    baserundir=f"runs/{sweep_name}_fix",
                )
                if cmd not in existing_set:
                    result_lines.append(cmd)
                    existing_set.add(cmd)

    if len(result_lines) != expected_count:
        raise RuntimeError(
            f"Suite {sweep_name} produced {len(result_lines)} tasks, expected {expected_count}"
        )
    return result_lines


def prepare_super_sweep(
    source_results_dir: Path = Path("results"),
    output_results_dir: Path = Path("results"),
    include_dev_in_super_manifest: bool = False,
    selected_optimizer_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write all per-suite manifests and the combined super-sweep manifest.

    ``selected_optimizer_ids`` restricts every suite except BBSubset-dev to these optimizers.
    """
    source_results_dir = Path(source_results_dir)
    output_results_dir = Path(output_results_dir)

    all_super_lines: list[str] = []
    dev_isolated_tasks = 0
    all_suites_total_tasks = 0
    suites_summary: dict[str, dict[str, Any]] = {}

    for raw_spec in EXPECTED_SUITE_SPECS:
        sweep_name = raw_spec["sweep_name"]
        spec = dict(raw_spec)

        if selected_optimizer_ids is not None and sweep_name != DEV_SUITE:
            filtered_opts = [opt for opt in spec["optimizers"] if opt["id"] in selected_optimizer_ids]
            if not filtered_opts:
                continue
            spec["expected_count"] = spec["expected_count"] * len(filtered_opts) // len(spec["optimizers"])
            spec["optimizers"] = filtered_opts

        suite_lines = prepare_suite_tasks(spec, source_results_dir=source_results_dir)

        suite_out_dir = output_results_dir / f"{sweep_name}_fix"
        suite_out_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(suite_lines) + "\n"
        tasks_all_path = suite_out_dir / "tasks_all.txt"
        tasks_txt_path = suite_out_dir / "tasks.txt"
        tasks_all_path.write_text(content, encoding="utf-8")
        tasks_txt_path.write_text(content, encoding="utf-8")

        all_suites_total_tasks += len(suite_lines)
        if sweep_name == DEV_SUITE:
            dev_isolated_tasks = len(suite_lines)
            if include_dev_in_super_manifest:
                all_super_lines.extend(suite_lines)
        else:
            all_super_lines.extend(suite_lines)

        suites_summary[sweep_name] = {
            "count": len(suite_lines),
            "isolated_from_super": sweep_name == DEV_SUITE and not include_dev_in_super_manifest,
            "tasks_all_path": str(tasks_all_path),
            "tasks_txt_path": str(tasks_txt_path),
        }

    super_out_dir = output_results_dir / "sweep_bo_uq_alignment_super"
    super_out_dir.mkdir(parents=True, exist_ok=True)
    super_content = "\n".join(all_super_lines) + "\n"
    super_all_path = super_out_dir / "tasks_all.txt"
    super_txt_path = super_out_dir / "tasks.txt"
    super_all_path.write_text(super_content, encoding="utf-8")
    super_txt_path.write_text(super_content, encoding="utf-8")

    return {
        "dev_isolated_tasks": dev_isolated_tasks,
        "total_tasks": len(all_super_lines),
        "all_suites_total_tasks": all_suites_total_tasks,
        "super_tasks_all_path": str(super_all_path),
        "super_tasks_txt_path": str(super_txt_path),
        "suites": suites_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the BBSubset-dev and BO super-sweep task manifests.")
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=Path("results"),
        help="Directory with existing <sweep>/tasks_all.txt manifests to reuse (default: results).",
    )
    parser.add_argument(
        "--output-results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the <sweep>_fix and super-sweep manifests (default: results).",
    )
    parser.add_argument(
        "--include-dev-in-super",
        action="store_true",
        help="Also add the BBSubset-dev tasks to sweep_bo_uq_alignment_super/tasks_all.txt.",
    )
    parser.add_argument("--optimizers", nargs="+", default=None, help="Restrict to these optimizer IDs.")
    parser.add_argument(
        "--five-methods",
        action="store_true",
        help=f"Restrict to {', '.join(FIVE_METHODS)} (76,200 non-dev tasks).",
    )
    args = parser.parse_args()

    summary = prepare_super_sweep(
        source_results_dir=args.source_results_dir,
        output_results_dir=args.output_results_dir,
        include_dev_in_super_manifest=args.include_dev_in_super,
        selected_optimizer_ids=FIVE_METHODS if args.five_methods else args.optimizers,
    )
    print(f"Prepared {len(summary['suites'])} suite manifests:")
    for sweep_name, info in summary["suites"].items():
        tag = " (not in super-sweep)" if info["isolated_from_super"] else ""
        print(f"  {sweep_name}_fix: {info['count']} tasks{tag} -> {info['tasks_all_path']}")
    print(
        f"BBSubset-dev tasks: {summary['dev_isolated_tasks']}, "
        f"super-sweep tasks: {summary['total_tasks']} "
        f"(all suites: {summary['all_suites_total_tasks']}) -> {summary['super_tasks_all_path']}"
    )


if __name__ == "__main__":
    main()
