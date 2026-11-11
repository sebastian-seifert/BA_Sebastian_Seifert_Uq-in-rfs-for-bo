#!/usr/bin/env python3
"""Summarises the BBOB timing sweep (timing_*.json from `dyrf_bo.bo.timed_runner`).

Wall-clock and BO-overhead statistics per optimizer and dimension, plus slowdown relative to
a baseline optimizer, written to timing_summary.md and timing_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dyrf_bo.sweeps.timing_tasks import (
    TIMING_DIMENSIONS,
    TIMING_SWEEP_OUTPUT_DIR,
)


def load_timing_records(results_dir: str) -> list[dict[str, Any]]:
    """Loads and validates all timing_*.json artifacts from results_dir."""
    records: list[dict[str, Any]] = []
    p = Path(results_dir)
    if not p.exists():
        return records

    for json_path in sorted(p.glob("**/timing_*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            if data.get("exit_code") != 0:
                continue

            task_info = data.get("task_info", {})
            opt_id = task_info.get("optimizer_id")
            dim = task_info.get("dimension")
            seed = task_info.get("seed")

            if opt_id is None or dim is None:
                continue

            records.append(
                {
                    "path": str(json_path),
                    "optimizer_id": str(opt_id),
                    "dimension": int(dim),
                    "function_id": task_info.get("function_id"),
                    "seed": seed,
                    "exit_code": data.get("exit_code", 0),
                    "wallclock_seconds": float(data.get("wallclock_seconds", 0.0)),
                    "bo_overhead_seconds": float(data.get("bo_overhead_seconds", 0.0)),
                    "target_eval_seconds": float(data.get("target_eval_seconds", 0.0)),
                    "hardware": data.get("hardware", {}),
                }
            )
        except Exception:
            continue

    return records


def _percentile(values: list[float], p: float) -> float:
    """Computes percentile p (0..100) using linear interpolation."""
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


def compute_dimension_approach_aggregates(
    records: list[dict[str, Any]],
    metric_key: str = "wallclock_seconds",
) -> dict[str, dict[int, dict[str, float]]]:
    """Groups metric by optimizer and dimension, computing summary statistics."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for r in records:
        opt = r["optimizer_id"]
        dim = r["dimension"]
        val = r[metric_key]
        grouped.setdefault(opt, {}).setdefault(dim, []).append(val)

    results: dict[str, dict[int, dict[str, float]]] = {}
    for opt, dim_map in grouped.items():
        results[opt] = {}
        for dim, values in dim_map.items():
            s_vals = sorted(values)
            n = len(s_vals)
            mean_val = sum(s_vals) / n if n > 0 else 0.0
            variance = sum((x - mean_val) ** 2 for x in s_vals) / n if n > 0 else 0.0
            std_val = math.sqrt(variance)

            median_val = _percentile(s_vals, 50.0)
            q25 = _percentile(s_vals, 25.0)
            q75 = _percentile(s_vals, 75.0)
            iqr = q75 - q25

            results[opt][dim] = {
                "count": float(n),
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
            }

    return results


def compute_relative_slowdown(
    aggregates: dict[str, dict[int, dict[str, float]]],
    baseline_id: str = "SMAC3_HPOFacade_lcb",
    agg_type: str = "mean",
) -> dict[str, dict[int, float]]:
    """Calculates slowdown ratio (optimizer / baseline) per dimension."""
    slowdowns: dict[str, dict[int, float]] = {}
    baseline_data = aggregates.get(baseline_id, {})

    for opt, dim_map in aggregates.items():
        slowdowns[opt] = {}
        for dim, stats in dim_map.items():
            base_stat = baseline_data.get(dim)
            if base_stat and base_stat[agg_type] > 0:
                slowdowns[opt][dim] = stats[agg_type] / base_stat[agg_type]
            else:
                slowdowns[opt][dim] = 1.0

    return slowdowns


def format_markdown_table(
    aggregates: dict[str, dict[int, dict[str, float]]],
    dimensions: Sequence[int] = TIMING_DIMENSIONS,
    agg_format: str = "mean_std",
) -> str:
    """Formats summary aggregates into a Markdown table."""
    headers = ["Approach"] + [f"Dim {d}" for d in dimensions]
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for opt in sorted(aggregates.keys()):
        row = [opt]
        for d in dimensions:
            stats = aggregates[opt].get(d)
            if stats:
                if agg_format == "mean_std":
                    cell = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
                elif agg_format == "median_iqr":
                    cell = f"{stats['median']:.2f} [{stats['q25']:.2f} - {stats['q75']:.2f}]"
                else:
                    cell = f"{stats['mean']:.2f}"
            else:
                cell = "N/A"
            row.append(cell)
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def format_slowdown_markdown_table(
    slowdowns: dict[str, dict[int, float]],
    dimensions: Sequence[int] = TIMING_DIMENSIONS,
) -> str:
    """Formats relative slowdown factors into a Markdown table."""
    headers = ["Approach"] + [f"Dim {d}" for d in dimensions]
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for opt in sorted(slowdowns.keys()):
        row = [opt]
        for d in dimensions:
            ratio = slowdowns[opt].get(d)
            if ratio is not None:
                cell = f"{ratio:.2f}x"
            else:
                cell = "N/A"
            row.append(cell)
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def export_analysis_results(
    results_dir: str = TIMING_SWEEP_OUTPUT_DIR,
    output_dir: str | None = None,
    baseline_id: str = "SMAC3_HPOFacade_lcb",
    dimensions: Sequence[int] = TIMING_DIMENSIONS,
) -> None:
    """Loads timing files, computes statistics, and writes report files."""
    out_dir = Path(output_dir or os.path.join(results_dir, "analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_timing_records(results_dir)
    if not records:
        print(f"No valid timing_*.json records found in '{results_dir}'.")
        return

    wallclock_aggs = compute_dimension_approach_aggregates(records, metric_key="wallclock_seconds")
    overhead_aggs = compute_dimension_approach_aggregates(records, metric_key="bo_overhead_seconds")
    slowdowns = compute_relative_slowdown(wallclock_aggs, baseline_id=baseline_id, agg_type="mean")

    md_content = []
    md_content.append("# BBOB Dimensionality Timing Sweep Analysis\n")
    md_content.append(f"Total valid runs parsed: **{len(records)}**\n")
    md_content.append("## 1. Wall-Clock Duration (Mean ± Std in seconds)\n")
    md_content.append(format_markdown_table(wallclock_aggs, dimensions=dimensions, agg_format="mean_std"))
    md_content.append("\n## 2. Wall-Clock Duration (Median [Q25 - Q75] in seconds)\n")
    md_content.append(format_markdown_table(wallclock_aggs, dimensions=dimensions, agg_format="median_iqr"))
    md_content.append(f"\n## 3. Relative Slowdown Factor (Normalized to `{baseline_id}` = 1.00x)\n")
    md_content.append(format_slowdown_markdown_table(slowdowns, dimensions=dimensions))
    md_content.append("\n## 4. Algorithmic BO Overhead (Mean ± Std in seconds)\n")
    md_content.append(format_markdown_table(overhead_aggs, dimensions=dimensions, agg_format="mean_std"))

    summary_md_path = out_dir / "timing_summary.md"
    summary_md_path.write_text("\n".join(md_content), encoding="utf-8")

    summary_csv_path = out_dir / "timing_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "approach",
                "dimension",
                "count",
                "mean_wallclock",
                "std_wallclock",
                "median_wallclock",
                "iqr_wallclock",
                "relative_slowdown",
            ]
        )
        for opt in sorted(wallclock_aggs.keys()):
            for d in dimensions:
                stats = wallclock_aggs[opt].get(d, {})
                ratio = slowdowns.get(opt, {}).get(d, 1.0)
                writer.writerow(
                    [
                        opt,
                        d,
                        stats.get("count", 0),
                        f"{stats.get('mean', 0.0):.4f}",
                        f"{stats.get('std', 0.0):.4f}",
                        f"{stats.get('median', 0.0):.4f}",
                        f"{stats.get('iqr', 0.0):.4f}",
                        f"{ratio:.4f}",
                    ]
                )

    print(f"Analysis successfully exported to: {summary_md_path} and {summary_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BBOB Dimensionality Timing Sweep results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=TIMING_SWEEP_OUTPUT_DIR,
        help="Directory containing timing_*.json files.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save analysis reports.")
    parser.add_argument("--baseline", type=str, default="SMAC3_HPOFacade_lcb", help="Baseline optimizer ID.")

    args = parser.parse_args()
    export_analysis_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        baseline_id=args.baseline,
    )


if __name__ == "__main__":
    main()
