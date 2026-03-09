"""Tests for regime.py — 4f: compute_regime_stats()."""

import numpy as np

from automl.analysis.time_series.single_ts_analysis.regime import compute_regime_stats


class TestComputeRegimeStats:
    def test_no_regimes(self, sample_target):
        result = compute_regime_stats(sample_target)
        assert result["regime_count"] == 0
        assert np.isnan(result["levene_stat"])

    def test_with_regimes_keys(self, sample_target, sample_regimes, regime_names):
        result = compute_regime_stats(sample_target, sample_regimes, regime_names)
        assert result["regime_count"] == 2
        assert "regime_low_vol_n" in result
        assert "regime_high_vol_n" in result
        assert "regime_low_vol_mean" in result
        assert "regime_high_vol_mean" in result

    def test_fractions_sum_to_one(self, sample_target, sample_regimes, regime_names):
        result = compute_regime_stats(sample_target, sample_regimes, regime_names)
        total = result["regime_low_vol_frac"] + result["regime_high_vol_frac"]
        assert abs(total - 1.0) < 1e-10

    def test_levene_produces_values(self, sample_target, sample_regimes, regime_names):
        result = compute_regime_stats(sample_target, sample_regimes, regime_names)
        assert not np.isnan(result["levene_stat"])
        assert 0.0 <= result["levene_pval"] <= 1.0

    def test_numeric_regime_names_fallback(self, sample_target, sample_regimes):
        result = compute_regime_stats(sample_target, sample_regimes)
        assert "regime_0_n" in result
        assert "regime_1_n" in result
