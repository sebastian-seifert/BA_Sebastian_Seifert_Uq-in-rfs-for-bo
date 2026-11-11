"""Per-task reference bounds for the meta-HPO loss.

Takes the empirical min/max of the incumbent cost per dev task from a CARP-S ``logs.csv`` of the
BBSubset-dev sweep and writes them to ``reference_bounds.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dyrf_bo.benchmarks.bbsubset import CarpsBBSubsetRegistry


def canonical_dev_task_name(raw_task_id: str) -> str:
    """Map a CARP-S task_id such as 'blackbox/20/dev/bbob/2/12/0' to 'subset_bbob_2_12_0'."""
    if "dev/" in raw_task_id:
        sub = raw_task_id.split("dev/")[-1]
    elif "blackbox/" in raw_task_id:
        sub = raw_task_id.split("blackbox/")[-1]
    else:
        sub = raw_task_id

    normalized = "subset_" + sub.replace("/", "_")
    return normalized


def extract_dev_reference_bounds(
    logs_csv_path: str,
    output_json: str | None = None,
    max_quantile: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Bounds {'min', 'max', 'median'} of ``trial_value__cost_inc`` per working dev task.

    The upper bound is the ``max_quantile`` quantile (1.0 = maximum). Tasks missing from the logs get
    [0, 1]. Optionally written to ``output_json``.
    """
    p = Path(logs_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Logs file not found: {logs_csv_path}")

    usecols = ["task_id", "trial_value__cost_inc"]
    df_raw = pd.read_csv(logs_csv_path, usecols=lambda c: c in usecols or c == "n_trials")

    df_raw["canonical_task"] = df_raw["task_id"].apply(canonical_dev_task_name)

    working_dev = [t.split("/")[-1] for t in CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)]
    df = df_raw[df_raw["canonical_task"].isin(working_dev)].copy()

    # Fall back to matching on the last path component of task_id.
    if df.empty:
        df_raw["canonical_task"] = df_raw["task_id"].apply(lambda t: t.split("/")[-1])
        df = df_raw[df_raw["canonical_task"].isin(working_dev)].copy()

    bounds: dict[str, dict[str, float]] = {}

    for task_name in df["canonical_task"].unique():
        task_df = df[df["canonical_task"] == task_name]
        costs = task_df["trial_value__cost_inc"].dropna().to_numpy(dtype=float)
        valid = costs[np.isfinite(costs)]

        if len(valid) == 0:
            continue

        y_min = float(np.min(valid))
        if max_quantile >= 1.0:
            y_max = float(np.max(valid))
        else:
            y_max = float(np.quantile(valid, max_quantile))

        y_median = float(np.median(valid))

        if y_max <= y_min:
            y_max = y_min + 1.0

        bounds[task_name] = {
            "min": round(y_min, 6),
            "max": round(y_max, 6),
            "median": round(y_median, 6),
        }

    for t in working_dev:
        if t not in bounds:
            bounds[t] = {"min": 0.0, "max": 1.0, "median": 0.5}

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(bounds, f, indent=2)

    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-task reference bounds for the meta-HPO loss from a CARP-S logs.csv."
    )
    parser.add_argument(
        "--logs",
        "-l",
        default="results/bbsubset_dev_analysis/logs.csv",
        help="Path to logs.csv file (default: results/bbsubset_dev_analysis/logs.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/meta_smac_proximity_hpo/reference_bounds.json",
        help="Path to output reference_bounds.json (default: results/meta_smac_proximity_hpo/reference_bounds.json)",
    )
    parser.add_argument(
        "--quantile",
        "-q",
        type=float,
        default=1.0,
        help="Quantile for y_max (default: 1.0 = absolute max)",
    )
    args = parser.parse_args()

    bounds = extract_dev_reference_bounds(
        logs_csv_path=args.logs,
        output_json=args.output,
        max_quantile=args.quantile,
    )
    print(f"Extracted empirical reference bounds for {len(bounds)} dev tasks.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
