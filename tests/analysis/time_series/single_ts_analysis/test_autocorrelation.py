"""Tests for autocorrelation.py — 4b: compute_autocorrelation()."""

import numpy as np

from automl.analysis.time_series.single_ts_analysis.autocorrelation import (
    compute_autocorrelation,
)


class TestComputeAutocorrelation:
    def test_output_keys(self, sample_target):
        result = compute_autocorrelation(sample_target, sampling_freq=1, horizon=1)
        for lag in [1, 2, 5, 10, 20]:
            assert f"acf_lag{lag}" in result
        assert "ljungbox_stat" in result
        assert "ljungbox_pval" in result
        assert "acf_downsample_k" in result
        assert result["acf_error"] is False

    def test_downsampling_factor(self, sample_target):
        result = compute_autocorrelation(sample_target, sampling_freq=1, horizon=4)
        assert result["acf_downsample_k"] == 4
        assert result["acf_n"] == 50  # 200 // 4

    def test_autocorrelated_series_has_nonzero_acf(self, sample_target):
        result = compute_autocorrelation(sample_target, sampling_freq=1, horizon=1)
        assert abs(result["acf_lag1"]) > 0.05

    def test_short_series_graceful(self, short_series):
        result = compute_autocorrelation(short_series, sampling_freq=1, horizon=1)
        assert result["acf_error"] is True
        assert np.isnan(result["acf_lag1"])

    def test_heavy_downsampling_insufficient_data(self, sample_target):
        # k=100 -> 200/100 = 2 obs, well below min 50
        result = compute_autocorrelation(sample_target, sampling_freq=1, horizon=100)
        assert result["acf_error"] is True
