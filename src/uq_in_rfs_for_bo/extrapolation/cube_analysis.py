"""Per-run ECE, multi-level Winkler and E-AURC for all approaches, aggregated over the
(stratum x surrogate x dimension) cube.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    compute_multi_level_winkler,
    compute_regression_eaurc,
    compute_regression_ece,
)
from .uq_evaluator import APPROACH_NAMES

METRIC_COLUMNS = [
    "ece",
    "winkler_mean",
    "winkler_95",
    "winkler_90",
    "winkler_80",
    "winkler_50",
    "e_aurc",
    "aurc",
    "oracle_aurc",
    "random_aurc",
    "naurc",
]


def evaluate_single_run_df(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all 10 approaches across all 5 stratum scopes ('all', 0, 1, 2, 3) for a single run.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame representing a single experimental run, containing test predictions,
        intervals, uncertainties, and strata.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with 50 rows (10 approaches x 5 stratum scopes) and computed metrics.
    """
    dimension = int(df["dimension"].iloc[0]) if "dimension" in df else 2
    n_train = int(df["n_train"].iloc[0]) if "n_train" in df else 34
    function_name = str(df["function_name"].iloc[0]) if "function_name" in df else "sphere"
    surrogate = str(df["surrogate"].iloc[0]) if "surrogate" in df else "smac_default"
    seed = int(df["seed"].iloc[0]) if "seed" in df else 0

    scopes: list[str | int] = ["all", 0, 1, 2, 3]
    records: list[dict[str, Any]] = []

    for scope in scopes:
        if scope == "all":
            mask = np.ones(len(df), dtype=bool)
            scope_str = "all"
        else:
            mask = df["stratum"].to_numpy() == int(scope)
            scope_str = str(scope)

        if not np.any(mask):
            continue

        y_true = df["y_true"].to_numpy()[mask]
        y_hat_col = "y_hat" if "y_hat" in df else "y_pred"
        y_hat = df[y_hat_col].to_numpy()[mask]

        for app in APPROACH_NAMES:
            # Older parquet files may lack the suffixed interval columns.
            interval_dict = {}
            for sfx in ("50", "80", "90", "95"):
                lwr_col = f"int_lwr_{app}_{sfx}"
                upr_col = f"int_upr_{app}_{sfx}"
                if lwr_col in df and upr_col in df:
                    interval_dict[sfx] = (
                        df[lwr_col].to_numpy()[mask],
                        df[upr_col].to_numpy()[mask],
                    )
                elif sfx == "95" and f"int_lwr_{app}" in df and f"int_upr_{app}" in df:
                    interval_dict[sfx] = (
                        df[f"int_lwr_{app}"].to_numpy()[mask],
                        df[f"int_upr_{app}"].to_numpy()[mask],
                    )

            ece_res = compute_regression_ece(y_true, interval_dict) if interval_dict else {"ece": np.nan}

            winkler_res = (
                compute_multi_level_winkler(y_true, interval_dict)
                if interval_dict
                else {
                    "winkler_mean": np.nan,
                    "winkler_95": np.nan,
                    "winkler_90": np.nan,
                    "winkler_80": np.nan,
                    "winkler_50": np.nan,
                }
            )

            # E-AURC ranks points by the exploration half-width.
            u_col = f"u_{app}"
            if u_col in df:
                u_arr = df[u_col].to_numpy()[mask]
            elif f"sigma_{app}" in df:
                u_arr = df[f"sigma_{app}"].to_numpy()[mask]
            else:
                u_arr = np.zeros_like(y_hat)

            eaurc_res = compute_regression_eaurc(y_true, y_hat, u_arr)

            rec = {
                "dimension": dimension,
                "n_train": n_train,
                "function_name": function_name,
                "surrogate": surrogate,
                "seed": seed,
                "stratum_scope": scope_str,
                "approach": app,
                "ece": ece_res.get("ece", np.nan),
                "winkler_mean": winkler_res.get("winkler_mean", np.nan),
                "winkler_95": winkler_res.get("winkler_95", np.nan),
                "winkler_90": winkler_res.get("winkler_90", np.nan),
                "winkler_80": winkler_res.get("winkler_80", np.nan),
                "winkler_50": winkler_res.get("winkler_50", np.nan),
                "e_aurc": eaurc_res.get("e_aurc", np.nan),
                "aurc": eaurc_res.get("aurc", np.nan),
                "oracle_aurc": eaurc_res.get("oracle_aurc", np.nan),
                "random_aurc": eaurc_res.get("random_aurc", np.nan),
                "naurc": eaurc_res.get("naurc", np.nan),
            }
            records.append(rec)

    return pd.DataFrame(records)


def _aggregate_cube_slice(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Helper computing mean, std, sem, median, iqr, and run counts for a cube slice."""
    all_group_cols = group_cols + ["approach"]
    grouped = df.groupby(all_group_cols, observed=True)

    agg_dict: dict[str, Any] = {}
    for col in METRIC_COLUMNS:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")
        agg_dict[f"{col}_sem"] = (col, lambda s: float(s.sem()) if len(s) > 1 else 0.0)
        agg_dict[f"{col}_median"] = (col, "median")
        agg_dict[f"{col}_iqr"] = (
            col,
            lambda s: (
                float(np.percentile(s.dropna(), 75) - np.percentile(s.dropna(), 25))
                if len(s.dropna()) > 0
                else 0.0
            ),
        )

    agg_dict["n_runs"] = ("ece", "count")
    res = grouped.agg(**agg_dict).reset_index()
    return res


def aggregate_3d_cube(runs_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate per-run evaluations into the 8 canonical slices of the 3D OLAP cube.

    Slices:
    1. main_global: (stratum='all', surrogate='all', dimension='all')
    2. stratum_global: (stratum=s, surrogate='all', dimension='all')
    3. main_global_by_dimension: (stratum='all', surrogate='all', dimension=d)
    4. main_global_by_surrogate: (stratum='all', surrogate=rf, dimension='all')
    5. stratum_by_dimension: (stratum=s, surrogate='all', dimension=d)
    6. stratum_by_surrogate: (stratum=s, surrogate=rf, dimension='all')
    7. main_global_rf_by_dimension: (stratum='all', surrogate=rf, dimension=d)
    8. full_cube: (stratum=s, surrogate=rf, dimension=d)

    Parameters
    ----------
    runs_df : pd.DataFrame
        Concatenated tidy DataFrame of evaluate_single_run_df outputs across all runs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping view names to their aggregated DataFrames.
    """
    df_global = runs_df[runs_df["stratum_scope"] == "all"].copy()
    df_strata = runs_df[runs_df["stratum_scope"] != "all"].copy()

    views: dict[str, pd.DataFrame] = {
        "main_global": _aggregate_cube_slice(df_global, []),
        "stratum_global": _aggregate_cube_slice(df_strata, ["stratum_scope"]),
        "main_global_by_dimension": _aggregate_cube_slice(df_global, ["dimension"]),
        "main_global_by_surrogate": _aggregate_cube_slice(df_global, ["surrogate"]),
        "stratum_by_dimension": _aggregate_cube_slice(df_strata, ["stratum_scope", "dimension"]),
        "stratum_by_surrogate": _aggregate_cube_slice(df_strata, ["stratum_scope", "surrogate"]),
        "main_global_rf_by_dimension": _aggregate_cube_slice(df_global, ["surrogate", "dimension"]),
        "full_cube": _aggregate_cube_slice(df_strata, ["stratum_scope", "surrogate", "dimension"]),
    }

    return views
