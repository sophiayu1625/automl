"""Tests for gate4_ic.py."""

import numpy as np

from automl.feature_selection.gate4_ic import (
    compute_cross_sectional_ic,
    compute_ic_stats,
    apply_gate4_decision,
)


class TestCrossSectionalIC:
    def test_strong_feature_positive_ic(self, panel_target, strong_feature):
        ic = compute_cross_sectional_ic(panel_target, strong_feature)
        assert ic.dropna().mean() > 0.3

    def test_noise_near_zero_ic(self, panel_target, noise_feature):
        ic = compute_cross_sectional_ic(panel_target, noise_feature)
        assert abs(ic.dropna().mean()) < 0.15

    def test_length_matches_timestamps(self, panel_target, strong_feature):
        ic = compute_cross_sectional_ic(panel_target, strong_feature)
        assert len(ic) == len(panel_target)


class TestICStats:
    def test_output_keys(self, panel_target, strong_feature):
        ic = compute_cross_sectional_ic(panel_target, strong_feature)
        stats = compute_ic_stats(ic)
        for key in ["ic_mean", "ic_std", "icir", "ic_positive_frac", "t_stat_nw", "p_value_nw"]:
            assert key in stats

    def test_strong_feature_significant(self, panel_target, strong_feature):
        ic = compute_cross_sectional_ic(panel_target, strong_feature)
        stats = compute_ic_stats(ic)
        assert stats["t_stat_nw"] > 2.0
        assert stats["p_value_nw"] < 0.05


class TestGate4Decision:
    def test_strong_passes(self, panel_target, strong_feature):
        ic = compute_cross_sectional_ic(panel_target, strong_feature)
        stats = compute_ic_stats(ic)
        decision = apply_gate4_decision(stats)
        assert decision["passed"] == True

    def test_noise_fails(self, panel_target, noise_feature):
        ic = compute_cross_sectional_ic(panel_target, noise_feature)
        stats = compute_ic_stats(ic)
        decision = apply_gate4_decision(stats)
        assert decision["passed"] == False
