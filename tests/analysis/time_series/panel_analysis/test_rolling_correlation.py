"""Tests for rolling_correlation.py — 6c: compute_rolling_correlation()."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.panel_analysis.rolling_correlation import (
    compute_rolling_correlation,
)


class TestComputeRollingCorrelation:
    def test_returns_dataframe(self, panel_target_independent, series_regime):
        result = compute_rolling_correlation(
            panel_target_independent, series_regime, rolling_window="48h",
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns_match_regimes(self, panel_target_independent, series_regime):
        result = compute_rolling_correlation(
            panel_target_independent, series_regime, rolling_window="48h",
        )
        assert set(result.columns) == {"A", "B"}

    def test_common_factor_high_rolling_corr(self, panel_target_common_factor, series_regime):
        result = compute_rolling_correlation(
            panel_target_common_factor, series_regime, rolling_window="48h",
        )
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.mean() > 0.8

    def test_single_member_regime_excluded(self, panel_target_independent):
        """Regime with 1 member can't have pairwise correlation."""
        regime = pd.Series(
            ["X"] * 9 + ["Y"],
            index=panel_target_independent.columns,
        )
        result = compute_rolling_correlation(
            panel_target_independent, regime, rolling_window="48h",
        )
        assert "Y" not in result.columns
