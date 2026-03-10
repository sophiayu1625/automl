"""Tests for gate3_redundancy.py."""

import numpy as np

from automl.feature_selection.gate3_redundancy import apply_gate3


class TestGate3:
    def test_redundant_pair_detected(self, redundant_pair):
        f1, f2 = redundant_pair
        result = apply_gate3({"f1": f1, "f2": f2}, corr_threshold=0.8)
        assert len(result["high_corr_pairs"]) > 0
        assert "f1" in result["flagged_redundant"] or "f2" in result["flagged_redundant"]

    def test_independent_no_flag(self, strong_feature, noise_feature):
        result = apply_gate3(
            {"strong": strong_feature, "noise": noise_feature},
            corr_threshold=0.8,
        )
        assert len(result["high_corr_pairs"]) == 0

    def test_vif_scores_computed(self, strong_feature, noise_feature):
        result = apply_gate3({"a": strong_feature, "b": noise_feature})
        assert "a" in result["vif_scores"]
        assert "b" in result["vif_scores"]

    def test_corr_matrix_shape(self, strong_feature, noise_feature):
        result = apply_gate3({"a": strong_feature, "b": noise_feature})
        assert result["corr_matrix"].shape == (2, 2)
