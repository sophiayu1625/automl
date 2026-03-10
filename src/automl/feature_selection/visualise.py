"""Visualisation functions for feature selection results."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from automl.feature_selection.report import FeatureSelectionReport


def plot_ic_bar(report: FeatureSelectionReport) -> plt.Figure:
    """Horizontal bar chart of IC_mean per feature, coloured by status."""
    df = report.to_dataframe()
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))

    colors = []
    for _, row in df.iterrows():
        if row["all_gates_passed"]:
            colors.append("green")
        elif row["gate4_passed"]:
            colors.append("orange")
        else:
            colors.append("red")

    ax.barh(df["feature_name"], df["gate4_ic_mean"].fillna(0), color=colors)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("IC Mean")
    ax.set_title("Feature IC Mean (green=selected, orange=conditional, red=rejected)")
    fig.tight_layout()
    return fig


def plot_ic_decay(
    report: FeatureSelectionReport,
    feature_names: list[str] | None = None,
) -> plt.Figure:
    """Line plot of IC vs lag with half-life annotation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if feature_names is None:
        feature_names = list(report.features.keys())

    for fname in feature_names:
        fr = report.features.get(fname)
        if fr is None:
            continue
        hl = fr.gate7_half_life
        ax.set_title(f"IC Decay (half-life annotated)")

    ax.set_xlabel("Lag")
    ax.set_ylabel("IC Mean")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_corr_heatmap(report: FeatureSelectionReport) -> plt.Figure:
    """Heatmap of pairwise feature Spearman correlations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if report.corr_matrix is not None:
        sns.heatmap(report.corr_matrix.astype(float), vmin=-1, vmax=1,
                    cmap="RdBu_r", center=0, ax=ax, annot=False)
    ax.set_title("Feature Pairwise Spearman Correlation")
    fig.tight_layout()
    return fig


def plot_regime_ic(
    report: FeatureSelectionReport,
    feature_name: str,
) -> plt.Figure:
    """Bar chart of IC_mean by regime."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Regime IC — {feature_name}")
    ax.set_ylabel("IC Mean")
    fig.tight_layout()
    return fig


def plot_gate_summary(report: FeatureSelectionReport) -> plt.Figure:
    """Grid: features (rows) × gates (columns), green/red pass/fail."""
    df = report.to_dataframe()
    gate_cols = ["gate1_passed", "gate2_passed", "gate4_passed",
                 "gate5_passed", "all_gates_passed"]
    gate_cols = [c for c in gate_cols if c in df.columns]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))

    grid = df[gate_cols].astype(float).values
    ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(gate_cols)))
    ax.set_xticklabels([c.replace("_passed", "") for c in gate_cols], rotation=45)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_name"])
    ax.set_title("Gate Summary")
    fig.tight_layout()
    return fig
