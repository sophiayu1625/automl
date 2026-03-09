"""Tests for visualise.py — plotting functions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from automl.analysis.time_series.panel_analysis.correlation import compute_correlation_matrix
from automl.analysis.time_series.panel_analysis.pca import compute_pca_per_regime
from automl.analysis.time_series.panel_analysis.rolling_correlation import compute_rolling_correlation
from automl.analysis.time_series.panel_analysis.visualise import (
    plot_correlation_heatmap,
    plot_pca_scree,
    plot_rolling_correlation,
    plot_metrics_by_regime,
    plot_metrics_vs_metadata,
)


class TestPlotFunctions:
    def test_correlation_heatmap(self, panel_target_independent, series_regime):
        corr = compute_correlation_matrix(panel_target_independent, series_regime)
        fig = plot_correlation_heatmap(corr["corr_matrix"], series_regime)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_scree(self, panel_level, series_regime):
        pca = compute_pca_per_regime(panel_level, series_regime)
        fig = plot_pca_scree(pca)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rolling_correlation_plot(self, panel_target_independent, series_regime, macro_regime):
        rc = compute_rolling_correlation(
            panel_target_independent, series_regime, rolling_window="48h",
        )
        fig = plot_rolling_correlation(rc, macro_regime)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_metrics_by_regime(self, results_df, series_regime):
        fig = plot_metrics_by_regime(results_df, series_regime, ["hurst", "desc_std"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_metrics_vs_metadata(self, results_df, series_metadata):
        fig = plot_metrics_vs_metadata(
            results_df, series_metadata, ["hurst"], "maturity",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
