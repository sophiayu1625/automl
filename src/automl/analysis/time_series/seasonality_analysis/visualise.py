"""Visualisation functions for seasonality analysis."""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_grouped_means(
    grouped_stats: pd.DataFrame,
    title: str = "Grouped Means",
    regime: Optional[str] = None,
) -> plt.Figure:
    """Bar plot of group means with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = grouped_stats["group_label"].astype(str)
    means = grouped_stats["mean"]
    ci_lower = grouped_stats["ci_lower_95"]
    ci_upper = grouped_stats["ci_upper_95"]
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    ax.bar(labels, means, yerr=[yerr_lower, yerr_upper], capsize=4, alpha=0.7)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Group")
    ax.set_ylabel("Mean")
    suffix = f" (regime={regime})" if regime else ""
    ax.set_title(f"{title}{suffix}")
    fig.tight_layout()
    return fig


def plot_periodogram(
    periodogram_result: dict,
    sampling_freq: str = "1h",
) -> plt.Figure:
    """Plot dominant periods from periodogram analysis."""
    fig, ax = plt.subplots(figsize=(8, 4))

    periods = periodogram_result.get("dominant_periods", [])
    powers = periodogram_result.get("dominant_powers", [])

    if periods and powers:
        ax.bar(range(len(periods)), powers, tick_label=[f"{p:.1f}" for p in periods])
        ax.set_xlabel(f"Period (in {sampling_freq} units)")
        ax.set_ylabel("Power")

    ax.set_title("Periodogram — Dominant Cycles")
    fig.tight_layout()
    return fig


def plot_stability(
    stability_result: dict,
    candidate_name: str = "",
) -> plt.Figure:
    """Visualise sub-sample stability results."""
    fig, ax = plt.subplots(figsize=(6, 4))

    n_sig = stability_result.get("n_significant", 0)
    direction = stability_result.get("direction_consistent", False)
    stable = stability_result.get("stable", False)

    text = (
        f"Significant splits: {n_sig}\n"
        f"Direction consistent: {direction}\n"
        f"Stable: {stable}"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=14,
            verticalalignment="center", horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightgreen" if stable else "lightyellow"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Stability — {candidate_name}")
    fig.tight_layout()
    return fig


def plot_cross_series_distribution(
    consistency_result: dict,
    candidate_name: str = "",
) -> plt.Figure:
    """Visualise cross-series consistency."""
    fig, ax = plt.subplots(figsize=(6, 4))

    pct_sig = consistency_result.get("pct_significant", 0)
    pct_dir = consistency_result.get("pct_same_direction", 0)
    consistent = consistency_result.get("consistent", False)

    bars = ax.bar(
        ["% Significant", "% Same Direction"],
        [pct_sig * 100 if not np.isnan(pct_sig) else 0,
         pct_dir * 100 if not np.isnan(pct_dir) else 0],
        color=["steelblue", "coral"],
    )
    ax.axhline(60, color="steelblue", linestyle="--", alpha=0.5, label="Sig threshold (60%)")
    ax.axhline(75, color="coral", linestyle="--", alpha=0.5, label="Dir threshold (75%)")
    ax.set_ylabel("Percentage")
    ax.set_title(f"Cross-Series Consistency — {candidate_name}\nConsistent: {consistent}")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return fig
