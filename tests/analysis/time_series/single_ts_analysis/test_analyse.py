"""Tests for analyse.py — main entry point: analyse_series()."""

import numpy as np

from automl.analysis.time_series.single_ts_analysis.analyse import analyse_series


class TestAnalyseSeries:
    def test_returns_flat_dict(self, sample_target, sample_level):
        result = analyse_series("data_A", sample_target, sample_level)
        assert isinstance(result, dict)
        assert result["series_id"] == "data_A"
        # All values should be scalars
        for v in result.values():
            assert not hasattr(v, "__len__") or isinstance(v, str)

    def test_contains_all_sections(self, sample_target, sample_level):
        result = analyse_series(
            "data_A", sample_target, sample_level,
            sampling_freq=1, horizon=1,
        )
        # 4a
        assert "desc_n" in result
        # 4b
        assert "acf_lag1" in result
        # 4c
        assert "adf_stat" in result
        # 4d
        assert "garch_omega" in result
        # 4e
        assert "rv_mean" in result
        # 4f
        assert "regime_count" in result

    def test_with_regimes(self, sample_target, sample_level, sample_regimes, regime_names):
        result = analyse_series(
            "data_B", sample_target, sample_level,
            regime_labels=sample_regimes,
            regime_names=regime_names,
        )
        assert "desc_rlow_vol_n" in result
        assert "regime_low_vol_n" in result

    def test_downsampling_params(self, sample_target, sample_level):
        result = analyse_series(
            "data_C", sample_target, sample_level,
            sampling_freq=1, horizon=4,
        )
        assert result["acf_downsample_k"] == 4
        assert result["stat_downsample_k"] == 4
