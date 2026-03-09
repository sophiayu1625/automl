"""Tests for descriptive.py — 4a: compute_descriptive_stats()."""

import numpy as np

from automl.analysis.time_series.single_ts_analysis.descriptive import (
    compute_descriptive_stats,
    _stats_for_array,
)


class TestStatsForArray:
    def test_basic_output_keys(self, rng):
        x = rng.normal(0, 1, 100)
        result = _stats_for_array(x, prefix="test")
        expected_keys = [
            "test_n", "test_mean", "test_std", "test_skew", "test_kurt",
            "test_p1", "test_p5", "test_p25", "test_p75", "test_p95", "test_p99",
        ]
        for k in expected_keys:
            assert k in result

    def test_insufficient_data(self):
        result = _stats_for_array(np.array([1.0]), prefix="x")
        assert np.isnan(result["x_mean"])

    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _stats_for_array(x, prefix="v")
        assert result["v_n"] == 5
        assert result["v_mean"] == 3.0


class TestComputeDescriptiveStats:
    def test_overall_stats(self, sample_target):
        result = compute_descriptive_stats(sample_target)
        assert "desc_n" in result
        assert result["desc_n"] == 200

    def test_with_regimes(self, sample_target, sample_regimes, regime_names):
        result = compute_descriptive_stats(sample_target, sample_regimes, regime_names)
        assert "desc_rlow_vol_n" in result
        assert "desc_rhigh_vol_n" in result
        assert result["desc_rlow_vol_n"] + result["desc_rhigh_vol_n"] == 200

    def test_without_regimes_no_regime_keys(self, sample_target):
        result = compute_descriptive_stats(sample_target)
        regime_keys = [k for k in result if k.startswith("desc_r")]
        assert len(regime_keys) == 0
