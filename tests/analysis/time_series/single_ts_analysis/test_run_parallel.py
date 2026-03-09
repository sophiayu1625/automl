"""Tests for run_parallel.py — parallel runner."""

import pandas as pd

from automl.analysis.time_series.single_ts_analysis.run_parallel import run_all


class TestRunAll:
    def test_returns_dataframe(self, sample_target, sample_level):
        inputs = [
            {"series_id": "s1", "target": sample_target, "level": sample_level},
            {"series_id": "s2", "target": sample_target, "level": sample_level},
        ]
        df = run_all(inputs, n_jobs=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df["series_id"]) == ["s1", "s2"]

    def test_single_series(self, sample_target, sample_level):
        inputs = [{"series_id": "only", "target": sample_target, "level": sample_level}]
        df = run_all(inputs, n_jobs=1)
        assert len(df) == 1

    def test_with_regimes(self, sample_target, sample_level, sample_regimes, regime_names):
        inputs = [
            {
                "series_id": "r1",
                "target": sample_target,
                "level": sample_level,
                "regime_labels": sample_regimes,
                "regime_names": regime_names,
            },
        ]
        df = run_all(inputs, n_jobs=1)
        assert "regime_low_vol_n" in df.columns
