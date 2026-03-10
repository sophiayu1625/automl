"""Tests for gate7_decay.py."""

import numpy as np

from automl.feature_selection.gate7_decay import compute_ic_decay


class TestGate7:
    def test_output_keys(self, panel_target, strong_feature):
        result = compute_ic_decay(panel_target, strong_feature, decay_lags=[1, 2, 5])
        assert "decay_curve" in result
        assert "tstat_curve" in result
        assert "half_life" in result
        assert "decay_classification" in result

    def test_decay_curve_lags(self, panel_target, strong_feature):
        lags = [1, 2, 5]
        result = compute_ic_decay(panel_target, strong_feature, decay_lags=lags)
        assert set(result["decay_curve"].keys()) == set(lags)

    def test_classification_values(self, panel_target, strong_feature):
        result = compute_ic_decay(panel_target, strong_feature)
        assert result["decay_classification"] in ["fast", "medium", "slow"]

    def test_noise_low_ic_all_lags(self, panel_target, noise_feature):
        result = compute_ic_decay(panel_target, noise_feature, decay_lags=[1, 2])
        for lag, ic_mean in result["decay_curve"].items():
            if not np.isnan(ic_mean):
                assert abs(ic_mean) < 0.2
