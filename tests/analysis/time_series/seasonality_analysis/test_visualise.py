"""Tests for visualise.py — plotting functions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from automl.analysis.time_series.seasonality_analysis.step1_visual import compute_grouped_stats
from automl.analysis.time_series.seasonality_analysis.step2_tests import compute_periodogram
from automl.analysis.time_series.seasonality_analysis.visualise import (
    plot_grouped_means,
    plot_periodogram,
    plot_stability,
    plot_cross_series_distribution,
)


class TestPlotFunctions:
    def test_grouped_means_plot(self, rng):
        series = pd.Series(rng.normal(0, 1, 200))
        groups = pd.Series(np.repeat([0, 1, 2, 3], 50))
        stats = compute_grouped_stats(series, groups)
        fig = plot_grouped_means(stats, title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_periodogram_plot(self, panel_white_noise):
        result = compute_periodogram(panel_white_noise)
        fig = plot_periodogram(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_stability_plot(self):
        result = {"n_significant": 2, "direction_consistent": True, "stable": True}
        fig = plot_stability(result, candidate_name="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cross_series_plot(self):
        result = {"pct_significant": 0.8, "pct_same_direction": 0.9, "consistent": True}
        fig = plot_cross_series_distribution(result, candidate_name="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
