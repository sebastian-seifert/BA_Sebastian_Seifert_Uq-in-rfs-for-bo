#!/usr/bin/env python3
"""Gather, merge and analyse BO sweep logs.

Per suite: gather `<sweep>_fix` run logs, merge them with the baseline rows of `<sweep>`
into `<sweep>_merged`, then optionally compute scorecards and CARP-S subset reports.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from dyrf_bo.analysis.batch_scorecard import process_single_log_file
from dyrf_bo.analysis.gather_bbob import gather_bbob_trial_logs
from dyrf_bo.analysis.gather_bbsubset import gather_trial_logs as gather_bbsubset_trial_logs
from dyrf_bo.analysis.gather_hpobench import gather_hpobench_ml_trial_logs
from dyrf_bo.paths import REPO_ROOT
from dyrf_bo.sweeps.bo_tasks import (
    DEV_PROXIMITY_OPTIMIZERS,
    OOB_PROXIMITY_OPTIMIZERS,
    SUITE_NAMES,
)

BASELINE_OPTIMIZER_IDS: list[str] = [
    "SMAC3_HPOFacade_lcb",
    "SMAC3_HPOFacade_ei",
    "SMAC20_Entropy_LCB",
]

OOB_REPORT_OPTIMIZERS: list[str] = BASELINE_OPTIMIZER_IDS + [opt["id"] for opt in OOB_PROXIMITY_OPTIMIZERS]

DEV_REPORT_OPTIMIZERS: list[str] = BASELINE_OPTIMIZER_IDS + [opt["id"] for opt in DEV_PROXIMITY_OPTIMIZERS]


def extract_yahpo_task_attributes(task_str: str, suite_name: str = "") -> dict[str, Any]:
    """Extracts canonical task_id, benchmark, dataset_id, and suite from a YAHPO task string."""
    cleaned = str(task_str).strip().rstrip("/")
    match = re.search(
        r"(?:cfg_rbv2_|rbv2[_/]|^)(ranger|super)[_/](\d+)(?:\D|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        variant = match.group(1).lower()
        dataset_id = int(match.group(2))
        task_id = f"cfg_rbv2_{variant}_{dataset_id}"
        benchmark = f"rbv2_{variant}"
        suite = f"yahpo_rbv2_{variant}"
    else:
        variant = (
            "ranger"
            if "ranger" in suite_name.lower()
            else ("super" if "super" in suite_name.lower() else "unknown")
        )
        dataset_id = -1
        task_id = cleaned.split("/")[-1] if "/" in cleaned else cleaned
        benchmark = f"rbv2_{variant}" if variant != "unknown" else "unknown"
        suite = f"yahpo_{benchmark}" if variant != "unknown" else "yahpo"

    return {
        "benchmark": benchmark,
        "dataset_id": dataset_id,
        "task_id": task_id,
        "suite": suite,
    }


def _parse_yahpo_run_dir_hierarchy(
    log_file: Path, runs_path: Path, suite_name: str = ""
) -> tuple[str, str, int]:
    """Parses (optimizer_id, task_id, seed) from CARP-S YAHPO directory hierarchy."""
    rel_parts = log_file.relative_to(runs_path).parts
    optimizer_id = rel_parts[0] if len(rel_parts) >= 1 else "unknown"

    if len(rel_parts) >= 4 and rel_parts[-3].lower() == "none":
        try:
            seed = int(rel_parts[-2])
        except ValueError:
            seed = -1
        raw_task = "/".join(rel_parts[1:-3])
    else:
        try:
            seed = int(rel_parts[-2])
            raw_task = "/".join(rel_parts[1:-2])
        except (ValueError, IndexError):
            raw_task = rel_parts[-2] if len(rel_parts) >= 2 else "unknown"
            seed = -1

    attrs = extract_yahpo_task_attributes(raw_task, suite_name=suite_name)
    return optimizer_id, attrs["task_id"], seed


def gather_yahpo_trial_logs(
    runs_base: Path | str,
    outdir: Path | str,
    suite_name: str = "sweep_yahpo_rbv2_ranger",
) -> pd.DataFrame:
    """Recursively parses YAHPO trial_logs.jsonl and fallback telemetry_*.json files."""
    runs_path = Path(runs_base)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    if runs_path.exists():
        jsonl_files = list(runs_path.glob("**/trial_logs.jsonl"))
        for log_file in jsonl_files:
            try:
                optimizer_id, task_id, seed = _parse_yahpo_run_dir_hierarchy(
                    log_file, runs_path, suite_name=suite_name
                )
                hydra_cfg = log_file.parent / ".hydra" / "config.yaml"
                if hydra_cfg.exists():
                    try:
                        import yaml

                        with open(hydra_cfg, encoding="utf-8") as hf:
                            cfg_data = yaml.safe_load(hf)
                        if isinstance(cfg_data, dict):
                            optimizer_id = cfg_data.get("optimizer_id", optimizer_id)
                            raw_t = cfg_data.get("task", {}).get("name", task_id)
                            task_id = extract_yahpo_task_attributes(raw_t, suite_name=suite_name)["task_id"]
                            seed = int(cfg_data.get("seed", seed))
                    except Exception:
                        pass

                task_attrs = extract_yahpo_task_attributes(task_id, suite_name=suite_name)

                with open(log_file, encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except Exception:
                            continue

                        trial = entry.get(
                            "trial",
                            entry.get(
                                "trial_idx",
                                entry.get("n_trials", entry.get("n_function_calls", line_idx)),
                            ),
                        )
                        val_dict = entry.get("trial_value") or {}
                        cost = val_dict.get("cost", None)
                        if cost is None:
                            cost = entry.get("cost", entry.get("trial_value__cost", None))

                        if cost is not None:
                            try:
                                cost = float(cost)
                            except (ValueError, TypeError):
                                cost = None

                        records.append(
                            {
                                "optimizer_id": optimizer_id,
                                "task_id": task_attrs["task_id"],
                                "seed": seed,
                                "trial": trial,
                                "n_trials": trial,
                                "cost": cost,
                                "trial_value__cost": cost,
                                "trial_value__cost_inc": None,  # filled below via cummin
                                **task_attrs,
                            }
                        )
            except Exception:
                continue

    # Fall back to telemetry_*.json files when no trial_logs.jsonl were found.
    if not records:
        search_paths = [out_path]
        if runs_path.exists() and runs_path != out_path:
            search_paths.append(runs_path)

        for s_path in search_paths:
            if not s_path.exists():
                continue
            for t_file in s_path.glob("**/telemetry_*.json"):
                try:
                    with open(t_file, encoding="utf-8") as f:
                        t_data = json.load(f)

                    opt_id = t_data.get("optimizer_id", "unknown")
                    raw_task = t_data.get(
                        "task_id",
                        t_data.get("task", t_data.get("task_name", "unknown")),
                    )
                    if raw_task == "unknown":
                        raw_task = t_file.stem.split("_seed")[0]

                    seed = int(t_data.get("seed", -1))
                    task_attrs = extract_yahpo_task_attributes(raw_task, suite_name=suite_name)

                    history = t_data.get("trials", t_data.get("history", []))
                    for h_idx, item in enumerate(history, 1):
                        c = item.get("cost", item.get("value", None))
                        trial = item.get("trial", item.get("trial_idx", item.get("n_trials", h_idx)))
                        if c is not None:
                            try:
                                c = float(c)
                            except (ValueError, TypeError):
                                c = None

                        records.append(
                            {
                                "optimizer_id": opt_id,
                                "task_id": task_attrs["task_id"],
                                "seed": seed,
                                "trial": trial,
                                "n_trials": trial,
                                "cost": c,
                                "trial_value__cost": c,
                                "trial_value__cost_inc": None,  # filled below via cummin
                                **task_attrs,
                            }
                        )
                except Exception:
                    continue

    if records:
        df = pd.DataFrame(records)
        grouper = ["task_id", "optimizer_id", "seed"]
        df = df.sort_values(grouper + ["trial"]).reset_index(drop=True)
        df["trial_value__cost_inc"] = df.groupby(grouper)["cost"].cummin()
    else:
        df = pd.DataFrame(
            columns=[
                "optimizer_id",
                "task_id",
                "seed",
                "trial",
                "n_trials",
                "cost",
                "trial_value__cost",
                "trial_value__cost_inc",
                "benchmark",
                "dataset_id",
                "suite",
            ]
        )

    parquet_path = out_path / "logs.parquet"
    csv_path = out_path / "logs.csv"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    df.to_csv(csv_path, index=False)
    return df


def gather_suite_fix_logs(
    suite_name: str,
    runs_dir: Path | str = Path("runs"),
    results_dir: Path | str = Path("results"),
) -> pd.DataFrame:
    """Gathers `<suite_name>_fix` trial logs and telemetry into `results/<suite_name>_fix/logs.parquet`."""
    if suite_name not in SUITE_NAMES:
        raise ValueError(f"Unknown suite '{suite_name}'. Expected one of: {SUITE_NAMES}")

    runs_fix_path = Path(runs_dir) / f"{suite_name}_fix"
    results_fix_path = Path(results_dir) / f"{suite_name}_fix"
    results_fix_path.mkdir(parents=True, exist_ok=True)

    if suite_name in ("sweep_bbsubset_dev_big_comparison", "sweep_bbsubset_test"):
        return gather_bbsubset_trial_logs(runs_fix_path, results_fix_path)

    if suite_name.startswith("sweep_bbob_dim"):
        dim = int(suite_name.replace("sweep_bbob_dim", ""))
        df = gather_bbob_trial_logs(runs_base=runs_fix_path, outdir=results_fix_path)
        if not df.empty and "dimension" in df.columns:
            df = df[df["dimension"] == dim].copy().reset_index(drop=True)
            try:
                df.to_parquet(results_fix_path / "logs.parquet", index=False)
            except Exception:
                pass
            df.to_csv(results_fix_path / "logs.csv", index=False)
        return df

    if suite_name in ("sweep_yahpo_rbv2_ranger", "sweep_yahpo_rbv2_super"):
        return gather_yahpo_trial_logs(
            runs_base=runs_fix_path,
            outdir=results_fix_path,
            suite_name=suite_name,
        )

    if suite_name == "sweep_hpobench_ml":
        return gather_hpobench_ml_trial_logs(
            runs_base=runs_fix_path,
            outdir=results_fix_path,
        )

    raise ValueError(f"Unhandled suite '{suite_name}'")


def _canonical_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames legacy `optimizer`/`task` columns to `optimizer_id`/`task_id`."""
    if "optimizer_id" not in df.columns and "optimizer" in df.columns:
        df = df.rename(columns={"optimizer": "optimizer_id"})
    if "task_id" not in df.columns and "task" in df.columns:
        df = df.rename(columns={"task": "task_id"})
    return df


def merge_suite_logs(old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merges non-proximity baseline rows from `old_df` with new rows from `new_df`."""
    new_working = _canonical_id_columns(new_df.copy())

    if old_df is None or old_df.empty:
        merged = new_working
    else:
        old_working = _canonical_id_columns(old_df.copy())

        new_opts = set(new_working["optimizer_id"].dropna().astype(str).unique().tolist())
        kept_old = old_working[
            (~old_working["optimizer_id"].astype(str).str.startswith("SMAC20_Proximity"))
            & (~old_working["optimizer_id"].astype(str).isin(new_opts))
        ].copy()

        merged = pd.concat([kept_old, new_working], ignore_index=True)

    sort_cols = [c for c in ["task_id", "optimizer_id", "seed", "trial", "n_trials"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    return merged


def validate_merged_logs(
    merged_df: pd.DataFrame,
    old_df: pd.DataFrame | None = None,
    strict: bool = True,
    new_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Checks that baseline rows survived the merge unchanged and that every
    (task_id, optimizer_id) pair has the same number of seeds."""
    errors: list[str] = []

    if merged_df.empty:
        errors.append("Merged DataFrame is empty.")
        if strict:
            raise ValueError(errors[0])
        return {"valid": False, "errors": errors}

    new_working: pd.DataFrame | None = None
    new_opts: set[str] = set()
    if new_df is not None and not new_df.empty:
        new_working = _canonical_id_columns(new_df.copy())
        if "optimizer_id" in new_working.columns:
            new_opts = set(new_working["optimizer_id"].dropna().astype(str).unique().tolist())

    # Baseline rows are compared against new_df if it re-ran that optimizer, otherwise against
    # old_df, but only when both share the same seed set.
    if old_df is not None and not old_df.empty:
        old_non_prox = old_df[~old_df["optimizer_id"].astype(str).str.startswith("SMAC20_Proximity")].copy()
        baseline_opts = sorted(old_non_prox["optimizer_id"].dropna().unique().tolist())

        merged_seeds = (
            set(merged_df["seed"].dropna().unique().tolist()) if "seed" in merged_df.columns else set()
        )
        old_seeds = (
            set(old_non_prox["seed"].dropna().unique().tolist()) if "seed" in old_non_prox.columns else set()
        )

        for base_opt in baseline_opts:
            if new_working is not None and base_opt in new_opts:
                ref_sub = new_working[new_working["optimizer_id"] == base_opt]
            elif merged_seeds == old_seeds:
                ref_sub = old_non_prox[old_non_prox["optimizer_id"] == base_opt]
            else:
                continue

            merged_sub = merged_df[merged_df["optimizer_id"] == base_opt]

            if len(ref_sub) != len(merged_sub):
                errors.append(
                    f"Baseline '{base_opt}' row count mismatch: expected {len(ref_sub)}, got {len(merged_sub)}"
                )
                continue

            common_cols = [c for c in ref_sub.columns if c in merged_sub.columns]
            sort_keys = [c for c in ["task_id", "seed", "trial", "n_trials"] if c in common_cols]
            ref_sorted = ref_sub[common_cols].sort_values(sort_keys).reset_index(drop=True)
            merged_sorted = merged_sub[common_cols].sort_values(sort_keys).reset_index(drop=True)

            try:
                pd.testing.assert_frame_equal(ref_sorted, merged_sorted, check_dtype=False, check_like=True)
            except AssertionError as exc:
                errors.append(f"Baseline '{base_opt}' rows were altered during merge: {exc}")

    if "seed" in merged_df.columns and "task_id" in merged_df.columns and "optimizer_id" in merged_df.columns:
        seed_counts = merged_df.groupby(["task_id", "optimizer_id"])["seed"].nunique()
        unique_tasks = sorted(merged_df["task_id"].dropna().unique().tolist())
        unique_opts = sorted(merged_df["optimizer_id"].dropna().unique().tolist())

        expected_pairs = len(unique_tasks) * len(unique_opts)
        if len(seed_counts) != expected_pairs:
            errors.append(
                f"Missing (task_id, optimizer_id) combinations in merged logs: "
                f"expected {expected_pairs} pairs across {len(unique_tasks)} tasks x {len(unique_opts)} optimizers, "
                f"got {len(seed_counts)} pairs (seed count mismatch)."
            )

        distinct_seed_counts = sorted(seed_counts.unique().tolist())
        if len(distinct_seed_counts) > 1:
            min_pair = seed_counts.idxmin()
            max_pair = seed_counts.idxmax()
            errors.append(
                f"Inconsistent seed count across (task_id, optimizer_id) pairs: "
                f"observed counts {distinct_seed_counts} (min={seed_counts.min()} at {min_pair}, "
                f"max={seed_counts.max()} at {max_pair})."
            )

    if errors and strict:
        raise ValueError("; ".join(errors))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_rows": len(merged_df),
        "n_tasks": int(merged_df["task_id"].nunique()) if "task_id" in merged_df.columns else 0,
        "optimizers": sorted(merged_df["optimizer_id"].dropna().unique().tolist())
        if "optimizer_id" in merged_df.columns
        else [],
    }


def run_suite_scorecard(
    merged_parquet_path: Path | str,
    subfolder: str = "new_table",
) -> dict[str, Any]:
    """Computes and exports custom scorecard (`new_table`) for a merged suite parquet file."""
    return process_single_log_file(
        input_path=Path(merged_parquet_path),
        subfolder=subfolder,
    )


def run_suite_carps_report(
    merged_parquet_path: Path | str,
    is_dev: bool = False,
    report_name: str = "subset_report",
    optimizers: list[str] | None = None,
    python_bin: str = sys.executable,
) -> int:
    """Runs `scripts/subset_report.py` on a merged suite parquet file."""
    parquet_path = Path(merged_parquet_path)
    target_optimizers = optimizers or (DEV_REPORT_OPTIMIZERS if is_dev else OOB_REPORT_OPTIMIZERS)
    script_path = REPO_ROOT / "scripts" / "subset_report.py"

    cmd = [
        python_bin,
        str(script_path),
        str(parquet_path),
        "--optimizers",
        *target_optimizers,
        "--report-name",
        report_name,
    ]
    res = subprocess.run(cmd, check=True)
    return int(res.returncode)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Gather, merge and analyse BO sweep logs (scorecards and CARP-S reports)."
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=SUITE_NAMES,
        help=f"Benchmark suites to process (default: all {len(SUITE_NAMES)} suites).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory containing <sweep>, <sweep>_fix, and <sweep>_merged (default: results).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Base runs directory containing <sweep>_fix execution folders (default: runs).",
    )
    parser.add_argument(
        "--gather",
        action="store_true",
        help="Gather trial logs and telemetry from runs/<sweep>_fix into results/<sweep>_fix/logs.parquet.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge non-proximity baseline rows from results/<sweep>/logs.parquet with results/<sweep>_fix/logs.parquet into results/<sweep>_merged/logs.parquet.",
    )
    parser.add_argument(
        "--run-scorecards",
        action="store_true",
        help="Run custom scorecard analysis (new_table) on results/<sweep>_merged/logs.parquet.",
    )
    parser.add_argument(
        "--run-carps",
        action="store_true",
        help="Run CARP-S subset report generation (reports_subset) on results/<sweep>_merged/logs.parquet.",
    )
    parser.add_argument(
        "--strict-validation",
        dest="strict_validation",
        action="store_true",
        default=True,
        help="Enforce strict baseline and seed-count validation when merging (default: True).",
    )
    parser.add_argument(
        "--no-strict-validation",
        dest="strict_validation",
        action="store_false",
        help="Log validation warnings without raising an exception on partial/in-progress sweeps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    runs_dir = Path(args.runs_dir)

    for suite in args.suites:
        if args.gather:
            gather_suite_fix_logs(suite, runs_dir=runs_dir, results_dir=results_dir)

        merged_dir = results_dir / f"{suite}_merged"
        merged_parquet = merged_dir / "logs.parquet"

        if args.merge:
            old_parquet = results_dir / suite / "logs.parquet"
            fix_parquet = results_dir / f"{suite}_fix" / "logs.parquet"

            if not fix_parquet.is_file():
                raise FileNotFoundError(f"Missing fix parquet file: {fix_parquet}")

            new_df = pd.read_parquet(fix_parquet)
            new_opt_col = "optimizer_id" if "optimizer_id" in new_df.columns else "optimizer"
            new_opts = (
                set(new_df[new_opt_col].dropna().astype(str).unique().tolist())
                if new_opt_col in new_df.columns
                else set()
            )
            has_all_baselines = set(BASELINE_OPTIMIZER_IDS).issubset(new_opts)

            if old_parquet.is_file():
                old_df: pd.DataFrame | None = pd.read_parquet(old_parquet)
            elif has_all_baselines:
                old_df = None
            else:
                raise FileNotFoundError(f"Missing baseline parquet file: {old_parquet}")

            merged_df = merge_suite_logs(old_df, new_df)
            validate_merged_logs(merged_df, old_df, strict=args.strict_validation, new_df=new_df)

            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(merged_parquet, index=False)
            merged_df.to_csv(merged_dir / "logs.csv", index=False)

        if args.run_scorecards:
            if not merged_parquet.is_file():
                raise FileNotFoundError(f"Cannot run scorecard; missing merged parquet: {merged_parquet}")
            outcome = run_suite_scorecard(merged_parquet)
            if outcome.get("status") != "success":
                return 1

        if args.run_carps:
            if not merged_parquet.is_file():
                raise FileNotFoundError(f"Cannot run CARP-S report; missing merged parquet: {merged_parquet}")
            is_dev = suite == "sweep_bbsubset_dev_big_comparison"
            rc = run_suite_carps_report(merged_parquet, is_dev=is_dev)
            if rc != 0:
                return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
