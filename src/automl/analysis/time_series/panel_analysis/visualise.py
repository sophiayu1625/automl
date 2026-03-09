"""Visualisation functions for panel analysis results."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    series_regime: pd.Series,
) -> plt.Figure:
    """Heatmap of pairwise correlations, ordered by regime."""
    order = series_regime.reindex(corr_matrix.index).sort_values().index.tolist()
    ordered = corr_matrix.loc[order, order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(ordered, vmin=-1, vmax=1, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Pairwise Correlation (ordered by regime)")
    fig.tight_layout()
    return fig


def plot_pca_scree(pca_results: dict) -> plt.Figure:
    """Scree plot of explained variance per regime."""
    n_regimes = len(pca_results)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 4), squeeze=False)

    for idx, (regime, res) in enumerate(sorted(pca_results.items())):
        ax = axes[0, idx]
        evr = res.get("explained_variance_ratio", np.array([]))
        if len(evr) == 0:
            ax.set_title(f"Regime {regime} (no data)")
            continue
        components = range(1, len(evr) + 1)
        ax.bar(components, evr, alpha=0.7, label="Individual")
        cum = res.get("cumulative_variance", np.cumsum(evr))
        ax.step(components, cum, where="mid", color="red", label="Cumulative")
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(f"Regime {regime}")
        ax.legend()

    fig.suptitle("PCA Scree Plot by Regime")
    fig.tight_layout()
    return fig


def plot_rolling_correlation(
    rolling_corr_df: pd.DataFrame,
    macro_regime: pd.Series | None = None,
) -> plt.Figure:
    """Time series of rolling mean correlations with optional regime shading."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in rolling_corr_df.columns:
        ax.plot(rolling_corr_df.index, rolling_corr_df[col], label=f"Regime {col}")

    if macro_regime is not None:
        aligned = macro_regime.reindex(rolling_corr_df.index).dropna()
        unique_mr = sorted(aligned.unique())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_mr)))
        for mr, color in zip(unique_mr, colors):
            mask = aligned == mr
            ax.fill_between(
                aligned.index, 0, 1,
                where=mask,
                alpha=0.2, color=color,
                transform=ax.get_xaxis_transform(),
                label=f"Macro {mr}",
            )

    ax.set_ylabel("Mean Pairwise Correlation")
    ax.set_title("Rolling Correlation by Regime")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_metrics_by_regime(
    results_df: pd.DataFrame,
    series_regime: pd.Series,
    metrics: list[str],
) -> plt.Figure:
    """Box plots of Phase 4 metrics grouped by regime."""
    df = results_df.copy()
    df["_regime"] = series_regime.reindex(df.index)
    df = df.dropna(subset=["_regime"])

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        if metric in df.columns:
            df.boxplot(column=metric, by="_regime", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Regime")

    fig.suptitle("Metrics by Regime", y=1.02)
    fig.tight_layout()
    return fig


def plot_metrics_vs_metadata(
    results_df: pd.DataFrame,
    series_metadata: pd.DataFrame,
    metrics: list[str],
    attr: str,
) -> plt.Figure:
    """Scatter plots of metrics versus a metadata attribute."""
    merged = results_df.join(series_metadata[[attr]], how="inner")

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        if metric in merged.columns and attr in merged.columns:
            valid = merged[[attr, metric]].dropna()
            ax.scatter(valid[attr], valid[metric], alpha=0.6)
            ax.set_xlabel(attr)
            ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs {attr}")

    fig.suptitle(f"Metrics vs {attr}", y=1.02)
    fig.tight_layout()
    return fig
