"""Tests for correlation.py — 6a: compute_correlation_matrix()."""

import numpy as np

from automl.analysis.time_series.panel_analysis.correlation import compute_correlation_matrix


class TestComputeCorrelationMatrix:
    def test_output_keys(self, panel_target_independent, series_regime):
        result = compute_correlation_matrix(panel_target_independent, series_regime)
        assert "corr_matrix" in result
        assert "within_regime_corr" in result
        assert "cross_regime_corr" in result

    def test_diagonal_is_one(self, panel_target_independent, series_regime):
        result = compute_correlation_matrix(panel_target_independent, series_regime)
        mat = result["corr_matrix"]
        for sid in mat.index:
            assert mat.loc[sid, sid] == 1.0

    def test_independent_series_low_corr(self, panel_target_independent, series_regime):
        result = compute_correlation_matrix(panel_target_independent, series_regime)
        mat = result["corr_matrix"]
        off_diag = mat.values[np.triu_indices_from(mat.values, k=1)]
        valid = off_diag[~np.isnan(off_diag)]
        assert np.abs(np.mean(valid)) < 0.2

    def test_common_factor_high_corr(self, panel_target_common_factor, series_regime):
        result = compute_correlation_matrix(panel_target_common_factor, series_regime)
        within = result["within_regime_corr"]
        for r, stats in within.items():
            if stats["n_pairs"] > 0:
                assert stats["mean"] > 0.9

    def test_min_overlap_filter(self, panel_target_independent, series_regime):
        # With very high min_overlap, some pairs should be NaN
        result = compute_correlation_matrix(
            panel_target_independent.iloc[:50], series_regime, min_overlap=100,
        )
        mat = result["corr_matrix"]
        off_diag = mat.values[np.triu_indices_from(mat.values, k=1)]
        assert np.all(np.isnan(off_diag))
