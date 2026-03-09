"""Tests for step3_stability.py — stability and cross-series consistency."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.seasonality_analysis.step3_stability import (
    check_stability,
    compute_cross_series_consistency,
)
from automl.analysis.time_series.seasonality_analysis.candidates import SeasonalCandidate


class TestStability:
    def test_stable_effect(self, panel_dow_effect, dow_candidate):
        # Pooled series with consistent DOW effect
        pooled = panel_dow_effect.stack()
        flat = pd.Series(pooled.values, index=pooled.index.get_level_values(0))
        grouping = dow_candidate.grouping_var.reindex(flat.index)
        result = check_stability(flat, grouping, dow_candidate, n_splits=3)
        assert result["stable"] is True

    def test_unstable_effect(self, panel_unstable_dow, dow_candidate):
        pooled = panel_unstable_dow.stack()
        flat = pd.Series(pooled.values, index=pooled.index.get_level_values(0))
        grouping = dow_candidate.grouping_var.reindex(flat.index)
        result = check_stability(flat, grouping, dow_candidate, n_splits=3)
        # Effect disappears in second half — should not be fully stable
        assert result["n_significant"] < 3


class TestCrossSeriesConsistency:
    def test_all_significant_same_direction(self):
        per_series = {
            f"s{i}": {"reject": True, "group_means": {"0": 2.0, "1": 0.0}}
            for i in range(10)
        }
        result = compute_cross_series_consistency(per_series)
        assert result["consistent"] is True
        assert result["pct_significant"] == 1.0

    def test_weak_consistency(self):
        per_series = {}
        for i in range(10):
            if i < 3:
                per_series[f"s{i}"] = {"reject": True, "group_means": {"0": 2.0, "1": 0.0}}
            else:
                per_series[f"s{i}"] = {"reject": False, "group_means": {"0": 0.1, "1": -0.1}}
        result = compute_cross_series_consistency(per_series)
        assert result["consistent"] is False
        assert result["pct_significant"] < 0.6

    def test_too_few_series(self):
        per_series = {f"s{i}": {"reject": True, "coef": 1.0} for i in range(3)}
        result = compute_cross_series_consistency(per_series)
        assert result["consistent"] is False
        assert np.isnan(result["pct_significant"])
