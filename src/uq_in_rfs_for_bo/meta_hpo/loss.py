"""Meta-HPO loss: mean normalized regret over the BBSubset dev tasks.

For task m with reference bounds [y_min_m, y_max_m], r_m = (mean_s y_m(s) - y_min_m) / (y_max_m - y_min_m);
tasks without any finite result get r_m = 1. The loss is the mean of r_m over all tasks with bounds.
"""

from __future__ import annotations

import json
import os

import numpy as np

from dyrf_bo.paths import REPO_ROOT

DEFAULT_BOUNDS_JSON = os.path.join(REPO_ROOT, "results", "meta_smac_proximity_hpo", "reference_bounds.json")


def load_dev_reference_bounds(bounds_file: str | None = None) -> dict[str, dict[str, float]]:
    """Reference bounds from ``bounds_file``; if missing, extract them from the dev-sweep logs.csv
    (and write the JSON), or return {} if that is missing too."""
    file_path = bounds_file or DEFAULT_BOUNDS_JSON

    if os.path.exists(file_path):
        with open(file_path) as f:
            return json.load(f)

    logs_csv = os.path.join(REPO_ROOT, "results", "bbsubset_dev_analysis", "logs.csv")
    if os.path.exists(logs_csv):
        from dyrf_bo.meta_hpo.reference_bounds import extract_dev_reference_bounds

        return extract_dev_reference_bounds(logs_csv, output_json=file_path)

    return {}


def compute_meta_loss(
    task_results: dict[str, list[float]],
    ref_bounds: dict[str, dict[str, float]] | None = None,
) -> float:
    """Meta-loss for per-task final costs (one per seed).

    ``ref_bounds`` maps task name to {'min', 'max'}; defaults to ``load_dev_reference_bounds()``.
    Returns 1.0 if no bounds are available.
    """
    if ref_bounds is None:
        ref_bounds = load_dev_reference_bounds()

    if not ref_bounds:
        return 1.0

    task_regrets: list[float] = []

    for task_name, bounds in ref_bounds.items():
        y_min = float(bounds.get("min", 0.0))
        y_max = float(bounds.get("max", 1.0))
        spread = y_max - y_min

        costs = task_results.get(task_name, [])
        valid_costs = [c for c in costs if np.isfinite(c)]

        if len(valid_costs) == 0:
            task_regrets.append(1.0)
            continue

        mean_cost = float(np.mean(valid_costs))

        if spread > 1e-12:
            norm_cost = (mean_cost - y_min) / spread
            task_regrets.append(norm_cost)
        else:
            task_regrets.append(0.0)

    return float(np.mean(task_regrets))
