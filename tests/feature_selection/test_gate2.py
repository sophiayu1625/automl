"""Tests for gate2_quality.py."""

import numpy as np

from automl.feature_selection.gate2_quality import apply_gate2


class TestGate2:
    def test_stationary_passes(self, strong_feature):
        result = apply_gate2({"feat": strong_feature})
        assert result["results"]["feat"]["coverage"] > 0.99
        # Stationary noise-like data should have high ADF reject frac
        assert result["results"]["feat"]["adf_reject_frac"] >= 0.7
        assert result["results"]["feat"]["passed"] == True

    def test_nonstationary_fails(self, nonstationary_feature):
        result = apply_gate2({"rw": nonstationary_feature})
        assert result["results"]["rw"]["adf_reject_frac"] < 0.7
        assert result["results"]["rw"]["passed"] == False

    def test_winsorised_panels_returned(self, strong_feature):
        result = apply_gate2({"feat": strong_feature})
        assert "feat" in result["winsorised_panels"]
        assert result["winsorised_panels"]["feat"].shape == strong_feature.shape

    def test_low_coverage_fails(self, strong_feature):
        sparse = strong_feature.copy()
        sparse.iloc[:400, :] = np.nan  # 80% missing
        result = apply_gate2({"sparse": sparse})
        assert result["results"]["sparse"]["coverage"] < 0.8
