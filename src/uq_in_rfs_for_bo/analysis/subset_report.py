#!/usr/bin/env python3
"""CARP-S report (ranks, critical-difference plots) for a subset of the optimizers in a logs.parquet.

Optimizers are selected by exact ID and/or regex pattern; `--inspect` lists the available ones.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def resolve_optimizer_column(df: pd.DataFrame) -> str:
    """Identify which column contains the optimizer identifier."""
    if "optimizer_id" in df.columns:
        return "optimizer_id"
    if "optimizer" in df.columns:
        return "optimizer"
    raise KeyError("Could not find 'optimizer_id' or 'optimizer' column in DataFrame.")


def filter_logs_dataframe(
    df: pd.DataFrame,
    optimizers: list[str] | None = None,
    patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter logs DataFrame to only include selected optimizers or matching patterns."""
    opt_col = resolve_optimizer_column(df)
    unique_opts = df[opt_col].unique().tolist()

    selected_set = set()

    if optimizers:
        for opt in optimizers:
            if opt in unique_opts:
                selected_set.add(opt)

    if patterns:
        for pat in patterns:
            for opt in unique_opts:
                if re.search(pat, str(opt), re.IGNORECASE):
                    selected_set.add(opt)

    if exclude_patterns:
        to_remove = set()
        for opt in selected_set:
            for expat in exclude_patterns:
                if re.search(expat, str(opt), re.IGNORECASE):
                    to_remove.add(opt)
        selected_set -= to_remove

    if not selected_set:
        raise ValueError(
            f"No matching optimizers found!\n"
            f"Requested exact: {optimizers}\n"
            f"Requested patterns: {patterns}\n"
            f"Excluded patterns: {exclude_patterns}\n"
            f"Available in logs: {unique_opts}"
        )

    filtered_df = df[df[opt_col].isin(selected_set)].copy()
    return filtered_df


def inspect_optimizers(parquet_path: Path) -> list[str]:
    """Inspect and return unique optimizer names in a parquet file."""
    df_sample = pd.read_parquet(parquet_path)
    opt_col = resolve_optimizer_column(df_sample)
    opts = sorted(df_sample[opt_col].unique().tolist())
    print(f"\nAvailable optimizers in {parquet_path} (column '{opt_col}', total {len(opts)}):")
    for opt in opts:
        print(f"  - {opt}")
    return opts


def ensure_report_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the metadata columns CARP-S `generate_report` expects if they are missing.

    `task_type` defaults to 'blackbox'; `task.optimization_resources.n_trials` (the budget)
    defaults to the per-task maximum of `n_trials`, or 100 if that cannot be computed.
    """
    df = df.copy()
    if "task_type" not in df.columns:
        df["task_type"] = "blackbox"
    if "task.optimization_resources.n_trials" not in df.columns:
        if "n_trials" in df.columns and "task_id" in df.columns:
            df["task.optimization_resources.n_trials"] = df.groupby("task_id")["n_trials"].transform("max")
        else:
            df["task.optimization_resources.n_trials"] = 100
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Filter CARPS logs.parquet and generate a focused subset report."
    )
    parser.add_argument(
        "result_path",
        type=str,
        help="Path to logs.parquet file.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        help="Exact optimizer_id names to include.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="Substring or regex patterns to match optimizer names (e.g. 'shaker_entropy' 'proximity_bc' 'HPOFacade').",
    )
    parser.add_argument(
        "--exclude",
        "--exclude-patterns",
        dest="exclude_patterns",
        nargs="+",
        default=None,
        help="Substring or regex patterns to exclude (e.g. 'proximity_bc' or '_pi_').",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save the generated report. Defaults to '<result_path_dir>/reports_subset'.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="subset_report",
        help="Subfolder name for the report.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable results normalization in generate_report.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="List all available optimizers in the parquet file and exit without generating a report.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.result_path)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}", file=sys.stderr)
        sys.exit(1)

    if args.inspect:
        inspect_optimizers(parquet_path)
        sys.exit(0)

    if not args.optimizers and not args.patterns:
        print(
            "Error: Please specify either --optimizers or --patterns to filter (or use --inspect to view options).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    opt_col = resolve_optimizer_column(df)
    filtered_df = filter_logs_dataframe(
        df,
        optimizers=args.optimizers,
        patterns=args.patterns,
        exclude_patterns=args.exclude_patterns,
    )
    matched_opts = filtered_df[opt_col].unique().tolist()
    print(f"Matched {len(matched_opts)} optimizers ({len(filtered_df)} / {len(df)} rows):")
    for opt in matched_opts:
        print(f"  * {opt}")

    report_dir = Path(args.report_dir) if args.report_dir else parquet_path.parent / "reports_subset"
    report_dir.mkdir(parents=True, exist_ok=True)

    filtered_df = ensure_report_compatibility(filtered_df)

    filtered_parquet_path = report_dir / f"{args.report_name}_filtered_logs.parquet"
    print(f"Writing filtered dataset to {filtered_parquet_path}...")
    filtered_df.to_parquet(filtered_parquet_path, index=False)

    print("\nRunning CARPS generate_report...")
    from carps.analysis.generate_report import generate_report

    generate_report(
        result_path=str(filtered_parquet_path),
        report_dir=report_dir,
        report_name=args.report_name,
        normalize_results=not args.no_normalize,
    )
    print(f"\nReport successfully generated in: {report_dir / args.report_name}")


if __name__ == "__main__":
    main()
