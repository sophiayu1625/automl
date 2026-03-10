"""Tests for run.py — run_feature_selection() integration."""

import pandas as pd

from automl.feature_selection.run import run_feature_selection
from automl.feature_selection.report import FeatureSelectionReport


class TestRunFeatureSelection:
    def test_strong_feature_selected(
        self, panel_target, strong_feature, series_regime, macro_regime,
    ):
        report = run_feature_selection(
            panel_target=panel_target,
            panel_features={"strong": strong_feature},
            series_regime=series_regime,
            macro_regime=macro_regime,
        )
        assert isinstance(report, FeatureSelectionReport)
        assert "strong" in report.selected_features

    def test_noise_rejected(
        self, panel_target, noise_feature, series_regime, macro_regime,
    ):
        report = run_feature_selection(
            panel_target=panel_target,
            panel_features={"noise": noise_feature},
            series_regime=series_regime,
            macro_regime=macro_regime,
        )
        assert "noise" in report.rejected_features

    def test_to_dataframe(
        self, panel_target, strong_feature, noise_feature, series_regime, macro_regime,
    ):
        report = run_feature_selection(
            panel_target=panel_target,
            panel_features={"strong": strong_feature, "noise": noise_feature},
            series_regime=series_regime,
            macro_regime=macro_regime,
        )
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_gate1_filter(
        self, panel_target, strong_feature, series_regime, macro_regime,
    ):
        report = run_feature_selection(
            panel_target=panel_target,
            panel_features={"strong": strong_feature},
            series_regime=series_regime,
            macro_regime=macro_regime,
            gate1_approved=[],  # reject all
        )
        assert "strong" in report.rejected_features
        assert report.features["strong"].rejection_reason == "gate1_economic_prior"

    def test_corr_matrix_in_report(
        self, panel_target, strong_feature, noise_feature, series_regime, macro_regime,
    ):
        report = run_feature_selection(
            panel_target=panel_target,
            panel_features={"a": strong_feature, "b": noise_feature},
            series_regime=series_regime,
            macro_regime=macro_regime,
        )
        assert report.corr_matrix is not None
        assert report.corr_matrix.shape == (2, 2)
