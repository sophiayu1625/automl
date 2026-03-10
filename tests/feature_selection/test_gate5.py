"""Tests for gate5_multiple.py."""

import numpy as np

from automl.feature_selection.gate5_multiple import apply_gate5


class TestGate5:
    def test_strong_feature_passes(self):
        stats = {
            "strong": {"p_value_nw": 0.001, "ic_mean": 0.1, "icir": 1.0, "t_stat_nw": 5.0},
        }
        result = apply_gate5(stats)
        assert result["strong"]["gate5_passed"] == True

    def test_noise_fails(self):
        stats = {
            "noise": {"p_value_nw": 0.8, "ic_mean": 0.001, "icir": 0.05, "t_stat_nw": 0.3},
        }
        result = apply_gate5(stats)
        assert result["noise"]["gate5_passed"] == False

    def test_bh_correction_with_many_noise(self, rng):
        """50 noise + 5 true — BH should select ~5."""
        stats = {}
        for i in range(50):
            stats[f"noise_{i}"] = {
                "p_value_nw": float(rng.uniform(0.3, 1.0)),
                "ic_mean": 0.001, "icir": 0.05, "t_stat_nw": 0.3,
            }
        for i in range(5):
            stats[f"true_{i}"] = {
                "p_value_nw": 0.001,
                "ic_mean": 0.1, "icir": 1.5, "t_stat_nw": 5.0,
            }
        result = apply_gate5(stats, fdr_q=0.10)
        n_passed = sum(1 for v in result.values() if v["gate5_passed"])
        assert 3 <= n_passed <= 10  # should be close to 5
