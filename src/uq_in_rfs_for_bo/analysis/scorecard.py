"""Scorecard of candidate optimizers against baseline controls on BO sweep logs.

Final incumbents are min-max scaled per task; per-task median regrets are compared with
win rates, one-sided Wilcoxon signed-rank tests (Holm-corrected) and rank-biserial correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

DEFAULT_CONTROLS: list[str] = ["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"]
DEFAULT_TIE_TOLERANCE: float = 1e-6
DEFAULT_COST_COL: str = "trial_value__cost_inc"


def apply_holm_bonferroni(p_vals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaN treated as 1.0)."""
    p = np.array(p_vals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    p = np.where(np.isnan(p), 1.0, p)
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    adj_sorted = np.zeros(n, dtype=float)
    for i in range(n):
        adj_sorted[i] = min(1.0, sorted_p[i] * (n - i))

    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])

    adj_original = np.zeros(n, dtype=float)
    adj_original[sort_idx] = adj_sorted
    return [float(val) for val in adj_original]


def calculate_rank_biserial_correlation(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    tie_tolerance: float = 1e-6,
) -> float:
    """Computes matched-pairs rank-biserial correlation r_rb = (W^+ - W^-) / (W^+ + W^-).

    Pairs with |x_i - y_i| <= tie_tolerance are dropped before ranking. Negative values
    mean x is lower (better for minimisation); 0.0 if all pairs are ties.
    """
    arr_x = np.asarray(x, dtype=float).ravel()
    arr_y = np.asarray(y, dtype=float).ravel()
    if len(arr_x) != len(arr_y) or len(arr_x) == 0:
        return 0.0

    valid = ~(np.isnan(arr_x) | np.isnan(arr_y))
    arr_x, arr_y = arr_x[valid], arr_y[valid]
    if len(arr_x) == 0:
        return 0.0

    diff = arr_x - arr_y
    nonzero = diff[np.abs(diff) > tie_tolerance]
    if len(nonzero) == 0:
        return 0.0

    abs_diff = np.abs(nonzero)
    ranks = rankdata(abs_diff)
    w_plus = float(np.sum(ranks[nonzero > 0]))  # Baseline wins (x > y)
    w_minus = float(np.sum(ranks[nonzero < 0]))  # Proposed wins (x < y)
    total = w_plus + w_minus
    if total <= 1e-12:
        return 0.0

    return float((w_plus - w_minus) / total)


def compute_min_max_scaled_regret(
    df: pd.DataFrame,
    cost_col: str = "trial_value__cost_inc",
) -> pd.DataFrame:
    """Adds `normalized_regret`: cost min-max scaled per task over all optimizers and seeds."""
    df_scaled = df.copy()
    if cost_col not in df_scaled.columns:
        for candidate in ["cost_inc", "trial_value__cost", "cost", "value"]:
            if candidate in df_scaled.columns:
                cost_col = candidate
                break

    task_bounds = df_scaled.groupby("task_id")[cost_col].agg(["min", "max"]).reset_index()
    task_bounds["spread"] = task_bounds["max"] - task_bounds["min"]
    merged = pd.merge(df_scaled, task_bounds, on="task_id", how="left")

    def _scale(row):
        spread = row["spread"]
        if spread > 1e-12:
            return (row[cost_col] - row["min"]) / spread
        return 0.0

    merged["normalized_regret"] = merged.apply(_scale, axis=1)
    return merged


def get_control_prefix(ctrl: str) -> str:
    """Maps a control optimizer ID to its column prefix (e.g. `ctrl_lcb`)."""
    ctrl_lower = ctrl.lower()
    if "lcb" in ctrl_lower:
        return "ctrl_lcb"
    elif "ei" in ctrl_lower:
        return "ctrl_ei"
    clean_name = ctrl.replace(" ", "_").replace("-", "_").lower()
    return f"ctrl_{clean_name}"


def compute_sweep_scorecard(
    df: pd.DataFrame,
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
    cost_col: str = "trial_value__cost_inc",
    tie_tolerance: float = 1e-6,
    holm_scope: str = "per_control",
) -> dict[str, Any]:
    """Computes the scorecard of all non-control optimizers against each control.

    Parameters
    ----------
    df : pd.DataFrame
        Optimization run logs containing at least task, optimizer, seed, and cost columns.
    controls : Sequence[str]
        Baseline optimizer IDs used as reference controls.
    cost_col : str
        Target cost column name.
    tie_tolerance : float
        Numerical threshold below which differences are treated as exact ties.
    holm_scope : str
        Holm-Bonferroni family: 'per_control' or 'joint' (all controls together).

    Returns
    -------
    dict with keys
        'scorecard': `optimizer_id` plus win rate, adjusted p-value and rank-biserial
                     correlation per control (controls are not rows).
        'full_scorecard': the same with run counts, regret means/medians and raw p-values.
        'scaled_df': final incumbents with min-max scaled regret.
    """
    working_df = df.copy()

    opt_col = "optimizer_id" if "optimizer_id" in working_df.columns else "optimizer"
    task_col = "task_id" if "task_id" in working_df.columns else "task"
    if opt_col not in working_df.columns or task_col not in working_df.columns:
        raise KeyError(
            f"DataFrame must contain optimizer and task columns. Found: {working_df.columns.tolist()}"
        )

    working_df = working_df.rename(columns={opt_col: "optimizer_id", task_col: "task_id"})

    if cost_col not in working_df.columns:
        for candidate in ["cost_inc", "trial_value__cost", "cost", "value"]:
            if candidate in working_df.columns:
                cost_col = candidate
                break

    # Final incumbent per run (task_id, optimizer_id, seed).
    seed_col = "seed" if "seed" in working_df.columns else None
    grouper = ["task_id", "optimizer_id"] + ([seed_col] if seed_col else [])
    trial_col = (
        "trial"
        if "trial" in working_df.columns
        else ("n_trials" if "n_trials" in working_df.columns else None)
    )

    if trial_col:
        idx = working_df.groupby(grouper)[trial_col].idxmax()
    else:
        idx = working_df.groupby(grouper)[cost_col].idxmin()
    final_df = working_df.loc[idx].copy()

    scaled_df = compute_min_max_scaled_regret(final_df, cost_col=cost_col)

    all_optimizers = sorted(scaled_df["optimizer_id"].dropna().unique())
    active_controls = [c for c in controls if c in all_optimizers]
    candidate_optimizers = [opt for opt in all_optimizers if opt not in active_controls]

    full_rows: list[dict[str, Any]] = []

    for opt in candidate_optimizers:
        opt_data = scaled_df[scaled_df["optimizer_id"] == opt]
        n_tasks = len(opt_data["task_id"].unique())
        n_runs = len(opt_data)
        mean_norm = float(opt_data["normalized_regret"].mean())
        median_norm = float(opt_data["normalized_regret"].median())

        row_full: dict[str, Any] = {
            "optimizer_id": opt,
            "n_tasks": n_tasks,
            "n_runs": n_runs,
            "mean_norm_regret": mean_norm,
            "median_norm_regret": median_norm,
        }

        opt_medians = opt_data.groupby("task_id")["normalized_regret"].median()

        for ctrl in active_controls:
            ctrl_prefix = get_control_prefix(ctrl)
            ctrl_data = scaled_df[scaled_df["optimizer_id"] == ctrl]
            ctrl_medians = ctrl_data.groupby("task_id")["normalized_regret"].median()

            common_tasks = sorted(set(opt_medians.index).intersection(set(ctrl_medians.index)))
            raw_opt = np.array([opt_medians[t] for t in common_tasks], dtype=float)
            raw_ctrl = np.array([ctrl_medians[t] for t in common_tasks], dtype=float)

            valid_mask = ~(np.isnan(raw_opt) | np.isnan(raw_ctrl))
            vec_opt = raw_opt[valid_mask]
            vec_ctrl = raw_ctrl[valid_mask]

            if len(vec_opt) == 0:
                row_full[f"{ctrl_prefix}_win_rate"] = np.nan
                row_full[f"{ctrl_prefix}_rank_biserial"] = np.nan
                row_full[f"{ctrl_prefix}_wilcoxon_p"] = np.nan
                continue

            diff = vec_opt - vec_ctrl
            wins = int(np.sum(diff < -tie_tolerance))
            losses = int(np.sum(diff > tie_tolerance))
            ties = len(diff) - wins - losses
            win_rate = (wins + 0.5 * ties) / len(diff)

            rb = calculate_rank_biserial_correlation(vec_opt, vec_ctrl, tie_tolerance=tie_tolerance)

            nonzeros = int(np.sum(np.abs(diff) > tie_tolerance))
            if nonzeros > 0:
                try:
                    res = wilcoxon(vec_opt, vec_ctrl, alternative="less", zero_method="wilcox")
                    p_val = float(res.pvalue)
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            row_full[f"{ctrl_prefix}_win_rate"] = float(win_rate)
            row_full[f"{ctrl_prefix}_rank_biserial"] = float(rb)
            row_full[f"{ctrl_prefix}_wilcoxon_p"] = float(p_val)

        full_rows.append(row_full)

    full_df = pd.DataFrame(full_rows)

    if not full_df.empty:
        if holm_scope == "joint":
            p_entries: list[tuple[float, int, str]] = []
            for ctrl in active_controls:
                ctrl_prefix = get_control_prefix(ctrl)
                p_col = f"{ctrl_prefix}_wilcoxon_p"
                adj_col = f"{ctrl_prefix}_wilcoxon_adj_p"
                full_df[adj_col] = np.nan
                if p_col in full_df.columns:
                    for idx_row, r_val in full_df.iterrows():
                        if pd.notna(r_val[p_col]):
                            p_entries.append((float(r_val[p_col]), idx_row, adj_col))
            if p_entries:
                adj_p_list = apply_holm_bonferroni([e[0] for e in p_entries])
                for (_, row_idx, adj_col), adj_val in zip(p_entries, adj_p_list):
                    full_df.loc[row_idx, adj_col] = adj_val
        else:
            for ctrl in active_controls:
                ctrl_prefix = get_control_prefix(ctrl)
                p_col = f"{ctrl_prefix}_wilcoxon_p"
                adj_col = f"{ctrl_prefix}_wilcoxon_adj_p"
                full_df[adj_col] = np.nan
                if p_col in full_df.columns:
                    mask = full_df[p_col].notna()
                    if mask.any():
                        adj_p = apply_holm_bonferroni(full_df.loc[mask, p_col].tolist())
                        full_df.loc[mask, adj_col] = adj_p

        if f"{get_control_prefix(active_controls[0])}_wilcoxon_adj_p" in full_df.columns:
            first_ctrl_prefix = get_control_prefix(active_controls[0])
            full_df = full_df.sort_values(
                [f"{first_ctrl_prefix}_wilcoxon_adj_p", "mean_norm_regret"]
            ).reset_index(drop=True)
        else:
            full_df = full_df.sort_values("mean_norm_regret").reset_index(drop=True)

    final_cols = ["optimizer_id"]
    for ctrl in active_controls:
        ctrl_prefix = get_control_prefix(ctrl)
        final_cols.extend(
            [
                f"{ctrl_prefix}_win_rate",
                f"{ctrl_prefix}_wilcoxon_adj_p",
                f"{ctrl_prefix}_rank_biserial",
            ]
        )

    available_final_cols = [c for c in final_cols if c in full_df.columns]
    scorecard_df = (
        full_df[available_final_cols].copy() if not full_df.empty else pd.DataFrame(columns=final_cols)
    )

    return {
        "scorecard": scorecard_df,
        "full_scorecard": full_df,
        "scaled_df": scaled_df,
    }


def format_scorecard_markdown(
    scorecard_df: pd.DataFrame,
    title: str = "Benchmark Sweep Scorecard",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> str:
    """Formats the scorecard as a Markdown document."""
    if scorecard_df.empty:
        return f"# {title}\n\n*No candidate approaches evaluated.*"

    display_df = scorecard_df.copy()

    rename_map: dict[str, str] = {"optimizer_id": "Optimizer"}
    for ctrl in controls:
        ctrl_prefix = get_control_prefix(ctrl)
        ctrl_short = "LCB" if "lcb" in ctrl.lower() else ("EI" if "ei" in ctrl.lower() else ctrl)
        rename_map[f"{ctrl_prefix}_win_rate"] = f"{ctrl_short} Win Rate"
        rename_map[f"{ctrl_prefix}_wilcoxon_adj_p"] = f"{ctrl_short} Adj. p"
        rename_map[f"{ctrl_prefix}_rank_biserial"] = f"{ctrl_short} $r_{{rb}}$"

    for col in display_df.columns:
        if col.endswith("_win_rate"):
            display_df[col] = display_df[col].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "N/A")
        elif col.endswith("_wilcoxon_adj_p"):
            display_df[col] = display_df[col].apply(
                lambda v: (
                    f"{v:.2e}" if (pd.notna(v) and v < 0.001) else (f"{v:.4f}" if pd.notna(v) else "N/A")
                )
            )
        elif col.endswith("_rank_biserial"):
            display_df[col] = display_df[col].apply(lambda v: f"{v:+.3f}" if pd.notna(v) else "N/A")

    display_df = display_df.rename(columns=rename_map)

    try:
        table_md = display_df.to_markdown(index=False)
    except Exception:
        table_md = display_df.to_string(index=False)

    lines = [
        f"# {title}",
        "",
        "**Statistical Notes**:",
        "- **Baselines**: Baseline optimizers serve solely as reference controls and are excluded from evaluated rows.",
        "- **Win Rate**: Proportion of benchmark tasks where candidate achieved lower normalized regret ($W + 0.5 T) / N$.",
        "- **Adj. p**: One-sided Wilcoxon signed-rank test ($H_1: \\text{cand} < \\text{baseline}$) with step-down Holm-Bonferroni correction.",
        "- **$r_{rb}$**: Matched-pairs rank-biserial correlation ($W^+ - W^-) / (W^+ + W^-$) with exact ties ($|\\Delta| \\le 10^{-6}$) strictly filtered out prior to ranking.",
        "",
        table_md,
        "",
    ]
    return "\n".join(lines)


def export_scorecard(
    results: dict[str, Any],
    output_dir: str | Path,
    title: str = "Benchmark Sweep Scorecard",
    controls: Sequence[str] = ("SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"),
) -> tuple[Path, Path]:
    """Writes scorecard.csv, scorecard_full_diagnostics.csv and scorecard.md to `output_dir`."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scorecard.csv"
    md_path = out_dir / "scorecard.md"

    results["scorecard"].to_csv(csv_path, index=False)

    if "full_scorecard" in results:
        results["full_scorecard"].to_csv(out_dir / "scorecard_full_diagnostics.csv", index=False)

    md_content = format_scorecard_markdown(results["scorecard"], title=title, controls=controls)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return csv_path, md_path


def load_logs_file(file_path: str | Path) -> pd.DataFrame:
    """Loads run logs from a Parquet or CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Logs file does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
