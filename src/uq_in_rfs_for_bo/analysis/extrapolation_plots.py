"""Plots of the extrapolation scorecards (global, per dimension, per stratum, per surrogate).

Writes two figure trees: ``focal_trio`` (Proximity BC (OOB), Shaker Entropy, Between-Tree
Disagreement) and ``with_variance_decomposition`` (adds Variance Decomposition). Error bars are 95% CIs.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

APPROACH_DISPLAY_NAMES: dict[str, str] = {
    "prox_bc_oob": "Proximity BC (OOB)",
    "prox_bc_cv": "Proximity BC (CV)",
    "prox_b_oob": "Proximity B (OOB)",
    "prox_b_cv": "Proximity B (CV)",
    "prox_ac_oob": "Proximity AC (OOB)",
    "prox_ac_cv": "Proximity AC (CV)",
    "prox_a_oob": "Proximity A (OOB)",
    "prox_a_cv": "Proximity A (CV)",
    "shaker_epistemic": "Shaker Entropy",
    "hutter_between": "Between-Tree Disagreement",
    "hutter_total": "Variance Decomposition",
}

FOCAL_TRIO: list[str] = ["prox_bc_oob", "shaker_epistemic", "hutter_between"]
QUARTET_WITH_TOTAL: list[str] = ["prox_bc_oob", "shaker_epistemic", "hutter_between", "hutter_total"]

APPROACH_STYLES: dict[str, dict[str, Any]] = {
    "prox_bc_oob": {"color": "#C2410C", "linestyle": "-", "marker": "o", "label": "Proximity BC (OOB)"},
    "shaker_epistemic": {"color": "#D97706", "linestyle": "-.", "marker": "^", "label": "Shaker Entropy"},
    "hutter_between": {
        "color": "#44403C",
        "linestyle": "--",
        "marker": "s",
        "label": "Between-Tree Disagreement",
    },
    "hutter_total": {"color": "#475569", "linestyle": ":", "marker": "D", "label": "Variance Decomposition"},
}

METRIC_SPECS = [
    {
        "stem": "ece",
        "mean_col": "ece_mean",
        "sem_col": "ece_sem",
        "title": "Expected Calibration Error (ECE)",
        "label": "ECE (lower is better)",
        "fmt": ".3f",
    },
    {
        "stem": "winkler_mean",
        "mean_col": "winkler_mean_mean",
        "sem_col": "winkler_mean_sem",
        "title": "Multi-Level Winkler Score",
        "label": "Winkler Score (lower is better)",
        "fmt": ".1f",
    },
    {
        "stem": "e_aurc",
        "mean_col": "e_aurc_mean",
        "sem_col": "e_aurc_sem",
        "title": "Excess AURC (E-AURC)",
        "label": "E-AURC (lower is better)",
        "fmt": ".3f",
    },
]

STRATUM_LABELS: dict[int, str] = {
    0: "0\n(Interpolation)",
    1: "1\n(Near)",
    2: "2\n(Moderate)",
    3: "3\n(Far)",
}

SURROGATE_ORDER: list[str] = ["smac_default", "mature", "shallow", "coarse"]
SURROGATE_DISPLAY_NAMES: dict[str, str] = {
    "smac_default": "SMAC Default",
    "mature": "Mature",
    "shallow": "Shallow",
    "coarse": "Coarse",
}

DEFAULT_FORMATS = ("pdf", "png")


def clean_approach_name(approach: str) -> str:
    return APPROACH_DISPLAY_NAMES.get(approach, approach)


def set_publication_rcparams() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
            ],
            "mathtext.fontset": "dejavusans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def get_warm_colorblind_palette(approaches: list[str]) -> list[str]:
    """Colours for the global point plots (proximity variants share the 'flare' palette)."""
    warm_colors = sns.color_palette("flare", n_colors=8).as_hex()
    proximity_color_map = {
        "prox_bc_oob": warm_colors[7],
        "prox_bc_cv": warm_colors[6],
        "prox_b_oob": warm_colors[5],
        "prox_b_cv": warm_colors[4],
        "prox_ac_oob": warm_colors[3],
        "prox_ac_cv": warm_colors[2],
        "prox_a_oob": warm_colors[1],
        "prox_a_cv": warm_colors[0],
    }
    fixed = {"hutter_between": "#44403C", "hutter_total": "#6B7280", "shaker_epistemic": "#D97706"}
    return [fixed.get(a) or proximity_color_map.get(a, warm_colors[4]) for a in approaches]


def _style(approach: str) -> dict[str, Any]:
    return APPROACH_STYLES.get(
        approach, {"color": "#333333", "linestyle": "-", "marker": "o", "label": approach}
    )


def _save_figure(fig, stem_path: Path, formats: Sequence[str], saved_files: list[Path]) -> None:
    for fmt in formats:
        p = stem_path.with_suffix(f".{fmt}")
        fig.savefig(p, bbox_inches="tight")
        saved_files.append(p)
    plt.close(fig)


def _plot_ci_lines(ax, df: pd.DataFrame, approaches: list[str], x_col: str, spec: dict[str, Any]) -> None:
    """Draw one line per approach with a shaded mean +/- 1.96 * SEM band."""
    for approach in approaches:
        sub = df[df["approach"] == approach].sort_values(x_col)
        if sub.empty:
            continue

        x = sub[x_col].values
        y = sub[spec["mean_col"]].values
        ci = sub[spec["sem_col"]].values * 1.96
        style = _style(approach)

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            facecolor=style["color"],
            edgecolor=style["color"],
            alpha=0.25,
            linewidth=0.6,
            zorder=2,
        )
        ax.plot(
            x,
            y,
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.2,
            markersize=5.0,
            markeredgecolor="#1F2937",
            markeredgewidth=0.5,
            zorder=3,
        )


def _has_metric(df: pd.DataFrame, spec: dict[str, Any]) -> bool:
    return spec["mean_col"] in df.columns and spec["sem_col"] in df.columns


def plot_global_point_plots(
    df: pd.DataFrame,
    output_dir: Path,
    exclude_approaches: list[str] | None = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    legend_loc: str = "upper right",
) -> list[Path]:
    """Horizontal point plots of the global metrics, best approach on top."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    if exclude_approaches is None:
        exclude_approaches = ["hutter_total"]

    df_plot = df[~df["approach"].isin(exclude_approaches)].copy()
    saved_files: list[Path] = []

    for spec in METRIC_SPECS:
        if not _has_metric(df_plot, spec):
            continue
        mean_col = spec["mean_col"]

        df_sorted = df_plot.sort_values(by=mean_col, ascending=False).reset_index(drop=True)

        fig_height = max(2.4, 0.38 * len(df_sorted))
        fig, ax = plt.subplots(figsize=(6.0, fig_height), dpi=300)

        y_pos = np.arange(len(df_sorted))
        means = df_sorted[mean_col].values
        ci_err = df_sorted[spec["sem_col"]].values * 1.96
        approaches = df_sorted["approach"].values
        labels = [clean_approach_name(a) for a in approaches]
        colors = get_warm_colorblind_palette(list(approaches))

        hutter_mask = df_sorted["approach"] == "hutter_between"
        if hutter_mask.any():
            hutter_val = df_sorted.loc[hutter_mask, mean_col].iloc[0]
            ax.axvline(
                x=hutter_val,
                color="#78716C",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label="Baseline",
                zorder=1,
            )
            ax.legend(
                loc=legend_loc,
                frameon=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#E5E7EB",
                fontsize=8.0,
            )

        for i in range(len(df_sorted)):
            ax.errorbar(
                x=means[i],
                y=y_pos[i],
                xerr=ci_err[i],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1.4,
                capsize=3.5,
                capthick=1.1,
                markersize=5.5,
                zorder=3,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(spec["label"])
        ax.set_title(f"Global {spec['title']}", pad=10)

        sns.despine(ax=ax, left=False, bottom=False)
        ax.xaxis.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.03 * x_range, x_max + 0.03 * x_range)

        _save_figure(fig, output_dir / f"global_{spec['stem']}", formats, saved_files)

    return saved_files


def plot_dimension_scaling(
    df: pd.DataFrame,
    approaches: list[str],
    output_dir: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> list[Path]:
    """Metric vs. dimension (log2 axis), one line per approach."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    df_filtered = df[df["approach"].isin(approaches)].copy()
    dims = sorted(df_filtered["dimension"].unique())
    saved_files: list[Path] = []

    for spec in METRIC_SPECS:
        if not _has_metric(df_filtered, spec):
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
        _plot_ci_lines(ax, df_filtered, approaches, "dimension", spec)

        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.minorticks_off()

        ax.set_xlabel("Dimension D")
        ax.set_ylabel(spec["label"])
        ax.set_title(f"Dimension Scaling — {spec['title']}", pad=10)

        ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB")
        sns.despine(ax=ax)
        ax.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        _save_figure(fig, output_dir / f"dimension_scaling_{spec['stem']}", formats, saved_files)

    return saved_files


def plot_stratum_scaling(
    df: pd.DataFrame,
    approaches: list[str],
    output_dir: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> list[Path]:
    """Metric vs. extrapolation stratum 0-3 (the 'all' scope is dropped)."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)

    df_filtered = df[df["stratum_scope"].isin([0, 1, 2, 3, "0", "1", "2", "3"])].copy()
    df_filtered["stratum"] = df_filtered["stratum_scope"].astype(int)
    df_filtered = df_filtered[df_filtered["approach"].isin(approaches)].copy()

    saved_files: list[Path] = []

    for spec in METRIC_SPECS:
        if not _has_metric(df_filtered, spec):
            continue

        fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
        _plot_ci_lines(ax, df_filtered, approaches, "stratum", spec)

        strata_ticks = [0, 1, 2, 3]
        ax.set_xticks(strata_ticks)
        ax.set_xticklabels([STRATUM_LABELS.get(s, str(s)) for s in strata_ticks])

        ax.set_xlabel("Extrapolation Stratum")
        ax.set_ylabel(spec["label"])
        ax.set_title(f"Stratum Scaling — {spec['title']}", pad=10)

        ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB")
        sns.despine(ax=ax)
        ax.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        _save_figure(fig, output_dir / f"stratum_scaling_{spec['stem']}", formats, saved_files)

    return saved_files


def plot_surrogate_scaling(
    df: pd.DataFrame,
    approaches: list[str],
    output_dir: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> list[Path]:
    """Dot-and-whisker chart per surrogate (rows), approaches dodged vertically."""
    set_publication_rcparams()
    output_dir.mkdir(parents=True, exist_ok=True)
    df_filtered = df[df["approach"].isin(approaches)].copy()

    surrogates = [s for s in SURROGATE_ORDER if s in df_filtered["surrogate"].unique()]
    if not surrogates:
        surrogates = sorted(df_filtered["surrogate"].unique())

    y_positions = np.arange(len(surrogates))

    if len(approaches) == 1:
        offsets = [0.0]
    elif len(approaches) == 2:
        offsets = [-0.11, 0.11]
    elif len(approaches) == 3:
        offsets = [-0.14, 0.0, 0.14]
    else:
        offsets = np.linspace(-0.16, 0.16, len(approaches)).tolist()
    offset_map = {app: offsets[idx] for idx, app in enumerate(approaches)}

    saved_files: list[Path] = []

    for spec in METRIC_SPECS:
        if not _has_metric(df_filtered, spec):
            continue
        mean_col = spec["mean_col"]
        sem_col = spec["sem_col"]

        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=300)

        for i in range(len(surrogates)):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, facecolor="#F9FAFB", edgecolor="none", zorder=0)

        for y_div in np.arange(0.5, len(surrogates) - 0.5, 1.0):
            ax.axhline(y_div, color="#E5E7EB", linewidth=0.6, alpha=0.6, zorder=0)

        for approach in approaches:
            style = _style(approach)
            y_offset = offset_map.get(approach, 0.0)
            x_vals = []
            ci_vals = []
            valid_y = []

            for i, s in enumerate(surrogates):
                sub = df_filtered[(df_filtered["surrogate"] == s) & (df_filtered["approach"] == approach)]
                if not sub.empty:
                    x_vals.append(sub[mean_col].values[0])
                    ci_vals.append(sub[sem_col].values[0] * 1.96)
                    valid_y.append(i + y_offset)

            if x_vals:
                ax.errorbar(
                    x=x_vals,
                    y=valid_y,
                    xerr=ci_vals,
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    label=style["label"],
                    markersize=5.5,
                    markeredgecolor="#1F2937",
                    markeredgewidth=0.7,
                    capsize=3.5,
                    capthick=0.9,
                    elinewidth=1.1,
                    linestyle="none",
                    zorder=3,
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([SURROGATE_DISPLAY_NAMES.get(s, s) for s in surrogates])
        # Inverted limits put the first surrogate (smac_default) at the top.
        ax.set_ylim(len(surrogates) - 0.5, -0.5)

        ax.set_xlabel(spec["label"])
        ax.set_title(f"Surrogate Scaling — {spec['title']}", pad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")

        ax.xaxis.grid(True, linestyle=":", alpha=0.4, color="#9CA3AF")
        ax.set_axisbelow(True)

        ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB", loc="best")

        _save_figure(fig, output_dir / f"surrogate_scaling_{spec['stem']}", formats, saved_files)

    return saved_files


def _plot_suite(
    output_dir: Path,
    approaches: list[str],
    exclude_global: list[str],
    df_main_global: pd.DataFrame,
    df_by_dimension: pd.DataFrame,
    df_stratum_global: pd.DataFrame,
    df_by_surrogate: pd.DataFrame,
    formats: Sequence[str],
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Render the four plot families for one approach set."""
    return (
        plot_global_point_plots(
            df=df_main_global,
            output_dir=output_dir / "global",
            exclude_approaches=exclude_global,
            formats=formats,
        ),
        plot_dimension_scaling(
            df=df_by_dimension, approaches=approaches, output_dir=output_dir / "dimension", formats=formats
        ),
        plot_stratum_scaling(
            df=df_stratum_global, approaches=approaches, output_dir=output_dir / "stratum", formats=formats
        ),
        plot_surrogate_scaling(
            df=df_by_surrogate, approaches=approaches, output_dir=output_dir / "surrogate", formats=formats
        ),
    )


def generate_all_extrapolation_plots(
    csv_dir: Path,
    output_dir: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> dict[str, list[Path]]:
    """Generate the focal_trio and with_variance_decomposition plot suites."""
    csv_main_global = csv_dir / "main_global.csv"
    csv_stratum_global = csv_dir / "stratum_global.csv"
    csv_by_dimension = csv_dir / "main_global_by_dimension.csv"
    csv_by_surrogate = csv_dir / "main_global_by_surrogate.csv"

    for req in [csv_main_global, csv_stratum_global, csv_by_dimension, csv_by_surrogate]:
        if not req.exists():
            raise FileNotFoundError(f"Required CSV not found: {req}")

    frames = dict(
        df_main_global=pd.read_csv(csv_main_global),
        df_stratum_global=pd.read_csv(csv_stratum_global),
        df_by_dimension=pd.read_csv(csv_by_dimension),
        df_by_surrogate=pd.read_csv(csv_by_surrogate),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[Path]] = {}

    trio_dir = output_dir / "focal_trio"
    trio_global, trio_dim, trio_stratum, trio_surr = _plot_suite(
        trio_dir, FOCAL_TRIO, ["hutter_total"], formats=formats, **frames
    )
    generated["focal_trio"] = trio_global + trio_dim + trio_stratum + trio_surr

    var_dir = output_dir / "with_variance_decomposition"
    var_global, var_dim, var_stratum, var_surr = _plot_suite(
        var_dir, QUARTET_WITH_TOTAL, [], formats=formats, **frames
    )
    generated["with_variance_decomposition"] = var_global + var_dim + var_stratum + var_surr

    # Mirror the focal_trio plots into an existing csv_dir/plots folder under the legacy file names.
    root_plots = csv_dir / "plots"
    if root_plots.exists():
        for p in trio_global:
            shutil.copy2(p, root_plots / p.name)
            (root_plots / "point_plots").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, root_plots / "point_plots" / f"point_ci_{p.name.replace('global_', '')}")
        for p in trio_surr:
            (root_plots / "surrogate_scaling").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, root_plots / "surrogate_scaling" / p.name.replace("scaling_", ""))
        for p in trio_dim:
            (root_plots / "dimension_scaling").mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                p, root_plots / "dimension_scaling" / p.name.replace("dimension_scaling_", "dim_scaling_")
            )
        for p in trio_stratum:
            (root_plots / "stratum_scaling").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, root_plots / "stratum_scaling" / p.name)

    print(f"[INFO] Generated {len(generated['focal_trio'])} files in {trio_dir}")
    print(f"[INFO] Generated {len(generated['with_variance_decomposition'])} files in {var_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade extrapolation UQ plots.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("results/extrapolation_uq/analysis"),
        help="Directory containing the 4 analysis CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extrapolation_uq/analysis/plots_updated"),
        help="Target output directory for the generated plots.",
    )

    args = parser.parse_args()
    generate_all_extrapolation_plots(csv_dir=args.csv_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
