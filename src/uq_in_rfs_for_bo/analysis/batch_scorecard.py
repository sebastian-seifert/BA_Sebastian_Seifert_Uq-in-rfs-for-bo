"""Runs the scorecard on a log file and writes the outputs next to it (`<parent>/new_table`)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dyrf_bo.analysis.scorecard import (
    DEFAULT_CONTROLS,
    DEFAULT_COST_COL,
    DEFAULT_TIE_TOLERANCE,
    compute_sweep_scorecard,
    export_scorecard,
    load_logs_file,
)

logger = logging.getLogger("batch_scorecard")


def resolve_batch_output_dir(input_path: str | Path, subfolder: str = "new_table") -> Path:
    """Resolves output directory to `<input_parent>/<subfolder>`."""
    resolved_path = Path(input_path).resolve()
    return resolved_path.parent / subfolder


def process_single_log_file(
    input_path: str | Path,
    subfolder: str = "new_table",
    controls: Sequence[str] = DEFAULT_CONTROLS,
    cost_col: str = DEFAULT_COST_COL,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    holm_scope: str = "per_control",
    title: str | None = None,
) -> dict[str, Any]:
    """Processes a single log file and writes scorecard outputs to `<input_dir>/<subfolder>`."""
    file_path = Path(input_path).resolve()
    if not file_path.exists():
        msg = f"Input file not found: {file_path}"
        logger.error(msg)
        return {"status": "error", "path": str(file_path), "error": msg}

    output_dir = resolve_batch_output_dir(file_path, subfolder=subfolder)
    output_dir.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"Scorecard: {file_path.parent.name}"

    try:
        logger.info(f"Loading data from {file_path} ...")
        df = load_logs_file(file_path)

        logger.info(f"Computing scorecard for {file_path} ...")
        results = compute_sweep_scorecard(
            df=df,
            controls=controls,
            cost_col=cost_col,
            tie_tolerance=tie_tolerance,
            holm_scope=holm_scope,
        )

        csv_path, md_path = export_scorecard(
            results=results,
            output_dir=output_dir,
            title=title,
            controls=controls,
        )
        logger.info(f"Exported scorecard to {csv_path} and {md_path}")

        return {
            "status": "success",
            "path": str(file_path),
            "output_dir": str(output_dir),
            "csv_path": str(csv_path),
            "md_path": str(md_path),
            "results": results,
        }
    except Exception as e:
        logger.exception(f"Failed processing {file_path}: {e}")
        return {"status": "error", "path": str(file_path), "error": str(e)}
