"""Tests for step2_tests.py — F-test, regression, ACF, periodogram."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.seasonality_analysis.step2_tests import (
    check_categorical_seasonality,
    check_continuous_seasonality,
    check_acf_at_seasonal_lag,
    compute_periodogram,
)


class TestCategoricalSeasonality:
    def test_detects_strong_effect(self, rng):
        n = 500
        groups = pd.Series(np.tile([0, 1, 2, 3, 4], n // 5))
        values = pd.Series(rng.normal(0, 0.5, n))
        values[groups == 0] += 3.0  # strong Monday effect
        result = check_categorical_seasonality(values, groups)
        assert result["reject"] == True
        assert result["f_stat"] > 10

    def test_no_effect_white_noise(self, rng):
        n = 500
        groups = pd.Series(np.tile([0, 1, 2, 3, 4], n // 5))
        values = pd.Series(rng.normal(0, 1, n))
        result = check_categorical_seasonality(values, groups)
        # May or may not reject, but F-stat should be small
        assert result["f_stat"] < 10


class TestContinuousSeasonality:
    def test_detects_linear_trend(self, rng):
        x = pd.Series(np.linspace(0, 10, 500))
        y = pd.Series(2.0 * x.values + rng.normal(0, 0.5, 500))
        result = check_continuous_seasonality(y, x)
        assert result["reject"] is True
        assert abs(result["coef"] - 2.0) < 0.5

    def test_no_relationship(self, rng):
        x = pd.Series(rng.uniform(0, 10, 500))
        y = pd.Series(rng.normal(0, 1, 500))
        result = check_continuous_seasonality(y, x)
        assert result["r_squared"] < 0.05


class TestACFAtSeasonalLag:
    def test_periodic_series(self, rng):
        n = 500
        t = np.arange(n)
        series = pd.Series(np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.1, n))
        result = check_acf_at_seasonal_lag(series, seasonal_period=24)
        assert result["acf_at_period"] > 0.3
        assert result["acf_significant"] == True

    def test_white_noise(self, rng):
        series = pd.Series(rng.normal(0, 1, 500))
        result = check_acf_at_seasonal_lag(series, seasonal_period=24)
        assert abs(result["acf_at_period"]) < 0.2


class TestPeriodogram:
    def test_detects_embedded_cycle(self, rng):
        n = 500
        t = np.arange(n)
        panel = pd.DataFrame({
            "s0": np.sin(2 * np.pi * t / 22) + rng.normal(0, 0.1, n),
            "s1": np.sin(2 * np.pi * t / 22) + rng.normal(0, 0.1, n),
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))
        result = compute_periodogram(panel)
        # Period ~22 should be among dominant
        assert any(18 < p < 26 for p in result["dominant_periods"])

    def test_white_noise_no_dominant(self, rng):
        panel = pd.DataFrame(
            rng.normal(0, 1, (500, 3)),
            index=pd.date_range("2020-01-01", periods=500, freq="h"),
            columns=["a", "b", "c"],
        )
        result = compute_periodogram(panel)
        assert len(result["dominant_periods"]) <= 5
