"""Tests for volatility.py — 4d: compute_garch(), 4e: compute_realized_variance()."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.single_ts_analysis.volatility import (
    compute_garch,
    compute_realized_variance,
)


class TestComputeGarch:
    def test_output_keys(self, sample_target):
        result = compute_garch(sample_target)
        for key in ["garch_omega", "garch_alpha", "garch_beta", "garch_gamma",
                     "garch_persistence", "garch_model_type", "garch_converged"]:
            assert key in result

    def test_convergence_on_sufficient_data(self, sample_target):
        result = compute_garch(sample_target)
        assert result["garch_converged"] is True
        assert result["garch_error"] is False

    def test_persistence_in_valid_range(self, sample_target):
        result = compute_garch(sample_target)
        if result["garch_converged"]:
            assert 0 < result["garch_persistence"] < 2

    def test_short_series_graceful(self, short_series):
        result = compute_garch(short_series)
        assert result["garch_error"] is True
        assert result["garch_converged"] is False

    def test_constant_series_graceful(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        const = pd.Series(np.ones(100), index=idx)
        result = compute_garch(const)
        assert result["garch_error"] is True


class TestComputeRealizedVariance:
    def test_output_keys(self, sample_target):
        result = compute_realized_variance(sample_target, sampling_freq=1, horizon=4)
        for key in ["rv_mean", "rv_std", "rv_median", "rv_n"]:
            assert key in result
        assert result["rv_error"] is False

    def test_rv_positive(self, sample_target):
        result = compute_realized_variance(sample_target, sampling_freq=1, horizon=4)
        assert result["rv_mean"] > 0

    def test_short_series_graceful(self, short_series):
        result = compute_realized_variance(short_series, sampling_freq=1, horizon=100)
        assert result["rv_error"] is True
