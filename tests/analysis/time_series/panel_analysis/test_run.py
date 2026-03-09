"""Tests for run.py — main entry point: run_panel_analysis()."""

from automl.analysis.time_series.panel_analysis.run import run_panel_analysis


class TestRunPanelAnalysis:
    def test_output_top_level_keys(
        self, panel_target_independent, panel_level,
        series_regime, macro_regime, results_df, series_metadata,
    ):
        result = run_panel_analysis(
            panel_target=panel_target_independent,
            panel_level=panel_level,
            series_regime=series_regime,
            macro_regime=macro_regime,
            results_df=results_df,
            series_metadata=series_metadata,
        )
        expected_keys = [
            "correlation", "pca", "rolling_correlation",
            "cross_regime", "architecture_recommendation", "boundary",
        ]
        for k in expected_keys:
            assert k in result

    def test_architecture_recommendation_valid(
        self, panel_target_independent, panel_level,
        series_regime, macro_regime, results_df, series_metadata,
    ):
        result = run_panel_analysis(
            panel_target=panel_target_independent,
            panel_level=panel_level,
            series_regime=series_regime,
            macro_regime=macro_regime,
            results_df=results_df,
            series_metadata=series_metadata,
        )
        assert result["architecture_recommendation"] in [
            "separate_models",
            "single_model_with_regime_feature",
            "investigate_further",
        ]
