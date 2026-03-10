"""Tests for gate6_regime.py."""

from automl.feature_selection.gate6_regime import compute_regime_ic


class TestGate6:
    def test_unconditional_strong_feature(
        self, panel_target, strong_feature, series_regime, macro_regime,
    ):
        result = compute_regime_ic(
            panel_target, strong_feature, series_regime, macro_regime,
        )
        assert result["classification"] == "unconditional"
        assert result["sign_flip_across_series_regime"] == False

    def test_sign_flip_detected(
        self, panel_target, regime_flip_feature, series_regime, macro_regime,
    ):
        result = compute_regime_ic(
            panel_target, regime_flip_feature, series_regime, macro_regime,
        )
        assert result["sign_flip_across_series_regime"] == True
        assert result["classification"] in ["series_conditional", "dangerous"]

    def test_output_keys(
        self, panel_target, strong_feature, series_regime, macro_regime,
    ):
        result = compute_regime_ic(
            panel_target, strong_feature, series_regime, macro_regime,
        )
        assert "by_series_regime" in result
        assert "by_macro_regime" in result
        assert "regime_stable" in result
