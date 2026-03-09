"""Tests for cross_regime.py — 6d: compare_regimes()."""

import numpy as np

from automl.analysis.time_series.panel_analysis.cross_regime import compare_regimes


class TestCompareRegimes:
    def test_output_keys(self, panel_target_independent, series_regime, results_df):
        result = compare_regimes(panel_target_independent, series_regime, results_df)
        assert "pairwise" in result
        assert "kruskal_wallis" in result
        assert "architecture_recommendation" in result

    def test_pairwise_contains_test_stats(self, panel_target_independent, series_regime, results_df):
        result = compare_regimes(panel_target_independent, series_regime, results_df)
        pair = result["pairwise"].get("A_vs_B", {})
        assert "levene_stat" in pair
        assert "ks_stat" in pair
        assert "ttest_stat" in pair

    def test_heteroscedastic_regimes_detect_difference(
        self, panel_target_heteroscedastic, series_regime, results_df,
    ):
        result = compare_regimes(panel_target_heteroscedastic, series_regime, results_df)
        pair = result["pairwise"]["A_vs_B"]
        # Levene should reject — very different variances
        assert pair["levene_reject"] == True

    def test_same_distribution_no_rejection(self, panel_target_independent, series_regime, results_df):
        result = compare_regimes(panel_target_independent, series_regime, results_df)
        pair = result["pairwise"]["A_vs_B"]
        # Independent N(0,1) vs N(0,1) — Levene should not reject
        assert pair["levene_reject"] == False

    def test_recommendation_values(self, panel_target_independent, series_regime, results_df):
        result = compare_regimes(panel_target_independent, series_regime, results_df)
        assert result["architecture_recommendation"] in [
            "separate_models",
            "single_model_with_regime_feature",
            "investigate_further",
        ]
