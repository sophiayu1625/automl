"""Tests for adjustment.py — seasonal component estimation."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.seasonality_analysis.adjustment import (
    estimate_seasonal_component,
)


class TestEstimateSeasonalComponent:
    def test_group_mean_removes_effect(self, rng):
        timestamps = pd.date_range("2020-01-01", periods=500, freq="h")
        groups = pd.Series(timestamps.dayofweek, index=timestamps)
        panel = pd.DataFrame({
            "s0": np.where(timestamps.dayofweek == 0, 2.0, 0.0) + rng.normal(0, 0.1, 500),
            "s1": np.where(timestamps.dayofweek == 0, 2.0, 0.0) + rng.normal(0, 0.1, 500),
        }, index=timestamps)

        seasonal = estimate_seasonal_component(panel, groups, "categorical", "group_mean")
        adjusted = panel - seasonal

        # Monday mean of adjusted should be close to zero
        monday_mean = adjusted.loc[timestamps.dayofweek == 0].mean().mean()
        other_mean = adjusted.loc[timestamps.dayofweek != 0].mean().mean()
        assert abs(monday_mean - other_mean) < 0.5

    def test_regression_method(self, rng):
        timestamps = pd.date_range("2020-01-01", periods=200, freq="h")
        x = pd.Series(np.linspace(0, 10, 200), index=timestamps)
        panel = pd.DataFrame({
            "s0": 3.0 * x.values + rng.normal(0, 0.1, 200),
        }, index=timestamps)

        seasonal = estimate_seasonal_component(panel, x, "continuous", "regression")
        assert seasonal.shape == panel.shape
        # Seasonal component should capture the linear trend
        assert seasonal["s0"].std() > 1.0

    def test_output_shape_matches_panel(self, rng):
        timestamps = pd.date_range("2020-01-01", periods=100, freq="h")
        groups = pd.Series(np.tile([0, 1], 50), index=timestamps)
        panel = pd.DataFrame(rng.normal(0, 1, (100, 3)), index=timestamps, columns=["a", "b", "c"])
        seasonal = estimate_seasonal_component(panel, groups, "categorical")
        assert seasonal.shape == panel.shape
