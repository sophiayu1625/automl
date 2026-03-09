"""Tests for boundary.py — 6e: analyse_boundary_effects()."""

import numpy as np

from automl.analysis.time_series.panel_analysis.boundary import analyse_boundary_effects


class TestAnalyseBoundaryEffects:
    def test_output_structure(self, results_df, series_metadata, series_regime):
        result = analyse_boundary_effects(results_df, series_metadata, series_regime)
        assert "A" in result
        assert "B" in result
        assert "distance_correlations" in result

    def test_per_regime_metric_keys(self, results_df, series_metadata, series_regime):
        result = analyse_boundary_effects(results_df, series_metadata, series_regime)
        for r in ["A", "B"]:
            for metric in ["hurst", "desc_std"]:
                if metric in result[r]:
                    entry = result[r][metric]
                    assert "mean_core" in entry
                    assert "mean_boundary" in entry
                    assert "mw_pvalue" in entry

    def test_missing_boundary_distance(self, results_df, series_regime):
        """Should return error if boundary_distance column missing."""
        import pandas as pd
        bad_meta = pd.DataFrame({"maturity": [1, 2]}, index=["s0", "s1"])
        result = analyse_boundary_effects(results_df, bad_meta, series_regime)
        assert "error" in result

    def test_distance_correlations_present(self, results_df, series_metadata, series_regime):
        result = analyse_boundary_effects(results_df, series_metadata, series_regime)
        dc = result["distance_correlations"]
        for metric in ["hurst", "desc_std"]:
            if metric in dc:
                assert "spearman_rho" in dc[metric]
                assert "spearman_pvalue" in dc[metric]
