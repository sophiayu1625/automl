"""Tests for visualise.py — plotting functions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automl.feature_selection.run import run_feature_selection
from automl.feature_selection.visualise import (
    plot_ic_bar,
    plot_corr_heatmap,
    plot_gate_summary,
)


class TestPlotFunctions:
    def _make_report(self, panel_target, strong_feature, noise_feature, series_regime, macro_regime):
        return run_feature_selection(
            panel_target=panel_target,
            panel_features={"strong": strong_feature, "noise": noise_feature},
            series_regime=series_regime,
            macro_regime=macro_regime,
        )

    def test_ic_bar(self, panel_target, strong_feature, noise_feature, series_regime, macro_regime):
        report = self._make_report(panel_target, strong_feature, noise_feature, series_regime, macro_regime)
        fig = plot_ic_bar(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_corr_heatmap(self, panel_target, strong_feature, noise_feature, series_regime, macro_regime):
        report = self._make_report(panel_target, strong_feature, noise_feature, series_regime, macro_regime)
        fig = plot_corr_heatmap(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gate_summary(self, panel_target, strong_feature, noise_feature, series_regime, macro_regime):
        report = self._make_report(panel_target, strong_feature, noise_feature, series_regime, macro_regime)
        fig = plot_gate_summary(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
