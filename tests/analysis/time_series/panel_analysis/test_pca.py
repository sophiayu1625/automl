"""Tests for pca.py — 6b: compute_pca_per_regime()."""

import numpy as np

from automl.analysis.time_series.panel_analysis.pca import compute_pca_per_regime


class TestComputePcaPerRegime:
    def test_output_keys_per_regime(self, panel_level, series_regime):
        result = compute_pca_per_regime(panel_level, series_regime)
        assert "A" in result
        assert "B" in result
        for r in result:
            assert "pc1_variance" in result[r]
            assert "explained_variance_ratio" in result[r]
            assert "loadings" in result[r]
            assert "n_series" in result[r]

    def test_common_factor_high_pc1(self, panel_target_common_factor, series_regime):
        level = panel_target_common_factor.cumsum()
        result = compute_pca_per_regime(level, series_regime)
        for r in result:
            if not result[r].get("error", False):
                assert result[r]["pc1_variance"] > 0.8

    def test_independent_series_lower_pc1(self, panel_level, series_regime):
        result = compute_pca_per_regime(panel_level, series_regime)
        for r in result:
            if not result[r].get("error", False):
                # Independent series shouldn't concentrate all variance in PC1
                assert result[r]["pc1_variance"] < 0.95

    def test_cumulative_variance_sums_correctly(self, panel_level, series_regime):
        result = compute_pca_per_regime(panel_level, series_regime)
        for r in result:
            if not result[r].get("error", False):
                evr = result[r]["explained_variance_ratio"]
                cum = result[r]["cumulative_variance"]
                np.testing.assert_allclose(cum, np.cumsum(evr))

    def test_few_series_graceful(self, series_regime):
        """Single-column regime should degrade gracefully."""
        import pandas as pd
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        panel = pd.DataFrame({"s0": np.random.randn(100)}, index=idx)
        regime = pd.Series(["X"], index=["s0"])
        result = compute_pca_per_regime(panel, regime)
        assert result["X"]["error"] is True
