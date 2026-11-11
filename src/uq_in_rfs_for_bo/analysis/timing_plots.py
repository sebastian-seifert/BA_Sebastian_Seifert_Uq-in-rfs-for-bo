#!/usr/bin/env python3
"""Runtime-scaling figures from timing_summary.csv.

timing_scaling_mean_std: wall-clock runtime (mean +/- SD) and slowdown vs. the LCB baseline per
optimizer; timing_scaling_by_family: the same averaged per optimizer family.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

DEFAULT_CSV_PATH = "results/sweep_bbob_timing/analysis/timing_summary.csv"
DEFAULT_OUTPUT_DIR = "results/sweep_bbob_timing/analysis"

FAMILY_ORDER = [
    "Baseline (LCB)",
    "Proximity Direct (A / B / AC / BC)",
    "Proximity CV (A / B / AC / BC)",
    "Shaker GMM Entropy",
]

FAMILY_STYLES: dict[str, dict[str, Any]] = {
    "Baseline (LCB)": {
        "color": "#2B2D42",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.0,
        "zorder": 3,
    },
    "Proximity Direct (A / B / AC / BC)": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
        "zorder": 4,
    },
    "Proximity CV (A / B / AC / BC)": {
        "color": "#009E73",
        "linestyle": "-.",
        "marker": "D",
        "linewidth": 2.2,
        "zorder": 4,
    },
    "Shaker GMM Entropy": {
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "^",
        "linewidth": 2.6,
        "zorder": 5,
    },
}

APPROACH_STYLES: dict[str, dict[str, Any]] = {
    "SMAC3_HPOFacade_lcb": {
        "label": "Baseline LCB (SMAC3)",
        "family": "Baseline (LCB)",
        "color": "#2B2D42",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.0,
        "offset": -0.075,
        "zorder": 5,
    },
    "SMAC3_HPOFacade_ei": {
        "label": "Baseline EI (SMAC3)",
        "family": None,  # not part of any family panel
        "color": "#6C757D",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.8,
        "offset": -0.060,
        "zorder": 4,
    },
    "SMAC20_Entropy_LCB": {
        "label": "Shaker GMM Entropy LCB",
        "family": "Shaker GMM Entropy",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "^",
        "linewidth": 2.6,
        "offset": 0.0,
        "zorder": 6,
    },
    "SMAC20_ProximityA_LCB": {
        "label": "Proximity-A LCB",
        "family": "Proximity Direct (A / B / AC / BC)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.9,
        "offset": -0.040,
        "zorder": 4,
    },
    "SMAC20_ProximityB_LCB": {
        "label": "Proximity-B LCB",
        "family": "Proximity Direct (A / B / AC / BC)",
        "color": "#56B4E9",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.9,
        "offset": -0.025,
        "zorder": 4,
    },
    "SMAC20_ProximityAC_LCB": {
        "label": "Proximity-AC LCB",
        "family": "Proximity Direct (A / B / AC / BC)",
        "color": "#332288",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.9,
        "offset": -0.010,
        "zorder": 4,
    },
    "SMAC20_ProximityBC_LCB": {
        "label": "Proximity-BC LCB",
        "family": "Proximity Direct (A / B / AC / BC)",
        "color": "#882255",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.9,
        "offset": 0.010,
        "zorder": 4,
    },
    "SMAC20_ProximityA_LCB_CV": {
        "label": "Proximity-A LCB (CV)",
        "family": "Proximity CV (A / B / AC / BC)",
        "color": "#009E73",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.9,
        "offset": 0.025,
        "zorder": 4,
    },
    "SMAC20_ProximityB_LCB_CV": {
        "label": "Proximity-B LCB (CV)",
        "family": "Proximity CV (A / B / AC / BC)",
        "color": "#44AA99",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.9,
        "offset": 0.040,
        "zorder": 4,
    },
    "SMAC20_ProximityAC_LCB_CV": {
        "label": "Proximity-AC LCB (CV)",
        "family": "Proximity CV (A / B / AC / BC)",
        "color": "#117733",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.9,
        "offset": 0.055,
        "zorder": 4,
    },
    "SMAC20_ProximityBC_LCB_CV": {
        "label": "Proximity-BC LCB (CV)",
        "family": "Proximity CV (A / B / AC / BC)",
        "color": "#999933",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.9,
        "offset": 0.070,
        "zorder": 4,
    },
}


def setup_plot_style() -> None:
    """Sets the shared matplotlib rcParams for the timing figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "savefig.dpi": 300,
        }
    )


def _format_log2_dim_axis(ax: plt.Axes, dims: list[int]) -> None:
    """Log2 x-axis with a tick at every dimension in `dims`."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(dims[0] * (2**-0.22), dims[-1] * (2**0.22))
    ax.set_xlabel("Problem Dimension $D$ ($\\log_2$ scale)")
    ax.grid(True, which="major", axis="both")


def plot_approach_scaling(df: pd.DataFrame, ax_a: plt.Axes, ax_b: plt.Axes) -> None:
    """Per-optimizer wall-clock runtime (ax_a) and slowdown relative to the baseline (ax_b)."""
    df = df.copy()
    df["dimension"] = df["dimension"].astype(int)
    dims = sorted(df["dimension"].unique().tolist())

    for app_id, style in APPROACH_STYLES.items():
        sub = df[df["approach"] == app_id].sort_values("dimension")
        if sub.empty:
            continue

        x_vals = sub["dimension"].to_numpy(dtype=float) * (2.0 ** style["offset"])
        y_mean = sub["mean_wallclock"].to_numpy(dtype=float)
        y_std = sub["std_wallclock"].to_numpy(dtype=float)
        rel_slowdown = sub["relative_slowdown"].to_numpy(dtype=float)

        ax_a.errorbar(
            x_vals,
            y_mean,
            yerr=y_std,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=5.5,
            capsize=3.5,
            elinewidth=1.0,
            alpha=0.92,
            zorder=style["zorder"],
        )

        ax_b.plot(
            x_vals,
            rel_slowdown,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=5.5,
            alpha=0.92,
            zorder=style["zorder"],
        )

    ax_b.axhline(1.0, color="#2B2D42", linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    _format_log2_dim_axis(ax_a, dims)
    _format_log2_dim_axis(ax_b, dims)

    ax_a.set_ylabel("Wall-Clock Runtime (s, Mean $\\pm$ 1 SD)")
    ax_a.set_title("(a) Absolute Wall-Clock Runtime Across Dimensions", pad=10)
    ax_a.set_ylim(bottom=0)

    ax_b.set_ylabel("Relative Slowdown vs. Baseline LCB ($\\times$)")
    ax_b.set_title("(b) Relative Algorithmic Overhead (Baseline LCB = $1.0\\times$)", pad=10)
    ax_b.set_ylim(bottom=0)


def plot_family_scaling(df: pd.DataFrame, ax_a: plt.Axes, ax_b: plt.Axes) -> None:
    """Same as `plot_approach_scaling`, averaged over the optimizers of each family."""
    df = df.copy()
    df["dimension"] = df["dimension"].astype(int)
    dims = sorted(df["dimension"].unique().tolist())
    df["family"] = df["approach"].map(lambda a: APPROACH_STYLES.get(a, {}).get("family", "Other"))

    family_offsets = {
        "Baseline (LCB)": -0.03,
        "Proximity Direct (A / B / AC / BC)": -0.01,
        "Proximity CV (A / B / AC / BC)": 0.01,
        "Shaker GMM Entropy": 0.03,
    }

    for fam in FAMILY_ORDER:
        sub_f = df[df["family"] == fam]
        if sub_f.empty:
            continue

        fstyle = FAMILY_STYLES[fam]
        grouped = (
            sub_f.groupby("dimension", as_index=False)
            .agg(
                mean_wc=("mean_wallclock", "mean"),
                # Mean of the member SDs, not a pooled SD.
                std_wc=("std_wallclock", "mean"),
                mean_rel=("relative_slowdown", "mean"),
            )
            .sort_values("dimension")
        )

        x_base = grouped["dimension"].to_numpy(dtype=float)
        x_jit = x_base * (2.0 ** family_offsets.get(fam, 0.0))
        y_wc = grouped["mean_wc"].to_numpy(dtype=float)
        s_wc = grouped["std_wc"].to_numpy(dtype=float)
        y_rel = grouped["mean_rel"].to_numpy(dtype=float)

        ax_a.fill_between(
            x_base,
            np.maximum(0.0, y_wc - s_wc),
            y_wc + s_wc,
            color=fstyle["color"],
            alpha=0.14,
            zorder=fstyle["zorder"] - 1,
        )
        ax_a.errorbar(
            x_jit,
            y_wc,
            yerr=s_wc,
            label=fam,
            color=fstyle["color"],
            linestyle=fstyle["linestyle"],
            marker=fstyle["marker"],
            linewidth=fstyle["linewidth"],
            markersize=6.0,
            capsize=3.5,
            elinewidth=1.1,
            zorder=fstyle["zorder"],
        )

        ax_b.plot(
            x_jit,
            y_rel,
            label=fam,
            color=fstyle["color"],
            linestyle=fstyle["linestyle"],
            marker=fstyle["marker"],
            linewidth=fstyle["linewidth"],
            markersize=6.0,
            alpha=0.95,
            zorder=fstyle["zorder"],
        )

    ax_b.axhline(1.0, color="#2B2D42", linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    _format_log2_dim_axis(ax_a, dims)
    _format_log2_dim_axis(ax_b, dims)

    ax_a.set_ylabel("Wall-Clock Runtime (s, Family Mean $\\pm$ 1 SD)")
    ax_a.set_title("(a) Architectural Family Runtime Scaling ($T=100$ Trials)", pad=10)
    ax_a.set_ylim(bottom=0)

    ax_b.set_ylabel("Relative Slowdown vs. Baseline LCB ($\\times$)")
    ax_b.set_title("(b) Architectural Family Relative Overhead", pad=10)
    ax_b.set_ylim(bottom=0)


def generate_timing_scaling_figures(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Writes timing_scaling_mean_std and timing_scaling_by_family (PDF and PNG) to `output_dir`."""
    setup_plot_style()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=False)
    fig1.subplots_adjust(top=0.76, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    plot_approach_scaling(df, ax1a, ax1b)

    handles, labels = ax1a.get_legend_handles_labels()
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        columnspacing=1.2,
        handletextpad=0.5,
    )

    mean_std_pdf = out_path / "timing_scaling_mean_std.pdf"
    mean_std_png = out_path / "timing_scaling_mean_std.png"
    fig1.savefig(mean_std_pdf, bbox_inches="tight")
    fig1.savefig(mean_std_png, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=False)
    fig2.subplots_adjust(top=0.81, bottom=0.15, left=0.07, right=0.98, wspace=0.22)
    plot_family_scaling(df, ax2a, ax2b)

    handles2, labels2 = ax2a.get_legend_handles_labels()
    fig2.legend(
        handles2,
        labels2,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.98),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        columnspacing=1.4,
    )

    family_pdf = out_path / "timing_scaling_by_family.pdf"
    family_png = out_path / "timing_scaling_by_family.png"
    fig2.savefig(family_pdf, bbox_inches="tight")
    fig2.savefig(family_png, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return {
        "mean_std_pdf": mean_std_pdf,
        "mean_std_png": mean_std_png,
        "family_pdf": family_pdf,
        "family_png": family_png,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BBOB timing-sweep runtime scaling over the problem dimension."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to timing_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output PDF and PNG figures.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Timing summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    outputs = generate_timing_scaling_figures(df, output_dir=args.output_dir)
    for k, v in outputs.items():
        print(f"Saved {k}: {v}")


if __name__ == "__main__":
    main()
