"""Tests for step1_visual.py — compute_grouped_stats()."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.seasonality_analysis.step1_visual import compute_grouped_stats


class TestComputeGroupedStats:
    def test_output_columns(self, rng):
        series = pd.Series(rng.normal(0, 1, 100))
        groups = pd.Series(np.repeat([0, 1, 2, 3], 25))
        result = compute_grouped_stats(series, groups)
        expected_cols = ["group_label", "mean", "std", "sem", "n", "ci_lower_95", "ci_upper_95"]
        for c in expected_cols:
            assert c in result.columns

    def test_ci_contains_mean(self, rng):
        series = pd.Series(rng.normal(0, 1, 200))
        groups = pd.Series(np.repeat([0, 1], 100))
        result = compute_grouped_stats(series, groups)
        for _, row in result.iterrows():
            assert row["ci_lower_95"] <= row["mean"] <= row["ci_upper_95"]

    def test_small_group_excluded(self, rng):
        series = pd.Series(rng.normal(0, 1, 103))
        groups = pd.Series([0] * 100 + [1] * 3)  # group 1 has < 5 obs
        result = compute_grouped_stats(series, groups)
        assert len(result) == 1  # only group 0
