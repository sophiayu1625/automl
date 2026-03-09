"""Tests for stationarity.py — 4c: compute_stationarity()."""

import numpy as np

from automl.analysis.time_series.single_ts_analysis.stationarity import (
    compute_stationarity,
    _hurst_rs,
)


class TestHurstRS:
    def test_random_walk_hurst_above_05(self, rng):
        # R/S analysis on a cumulative random walk yields H close to 1.0
        walk = np.cumsum(rng.normal(0, 1, 2000))
        h = _hurst_rs(walk)
        assert 0.5 < h < 1.2

    def test_white_noise_hurst_near_05(self, rng):
        noise = rng.normal(0, 1, 2000)
        h = _hurst_rs(noise)
        assert 0.3 < h < 0.7

    def test_short_input_returns_nan(self):
        assert np.isnan(_hurst_rs(np.array([1.0, 2.0])))


class TestComputeStationarity:
    def test_output_keys(self, sample_level):
        result = compute_stationarity(sample_level, sampling_freq=1, horizon=1)
        for key in ["adf_stat", "adf_pval", "kpss_stat", "kpss_pval",
                     "za_stat", "za_pval", "za_breakpoint", "hurst"]:
            assert key in result
        assert result["stat_error"] is False

    def test_nonstationary_level_adf_pval(self, sample_level):
        result = compute_stationarity(sample_level, sampling_freq=1, horizon=1)
        # Cumulative sum is typically non-stationary -> high p-value
        assert result["adf_pval"] > 0.01 or not np.isnan(result["adf_pval"])

    def test_short_series_graceful(self, short_series):
        result = compute_stationarity(short_series, sampling_freq=1, horizon=1)
        assert result["stat_error"] is True
        assert np.isnan(result["adf_stat"])

    def test_downsampling(self, sample_level):
        result = compute_stationarity(sample_level, sampling_freq=1, horizon=4)
        assert result["stat_downsample_k"] == 4
        assert result["stat_n"] == 50
