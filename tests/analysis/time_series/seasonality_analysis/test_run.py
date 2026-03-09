"""Tests for run.py — run_seasonality_analysis() integration."""

from automl.analysis.time_series.seasonality_analysis.run import (
    run_seasonality_analysis,
    SeasonalityReport,
)


class TestRunSeasonalityAnalysis:
    def test_dow_effect_validated(self, panel_dow_effect, series_regime, dow_candidate):
        report = run_seasonality_analysis(
            panel_target=panel_dow_effect,
            series_regime=series_regime,
            candidates=[dow_candidate],
        )
        assert isinstance(report, SeasonalityReport)
        assert "day_of_week" in report.candidates
        cr = report.candidates["day_of_week"]
        assert cr.validated is True
        assert cr.seasonal_component is not None
        assert cr.adjusted_panel_target is not None

    def test_white_noise_rejected(self, panel_white_noise, series_regime, dow_candidate):
        report = run_seasonality_analysis(
            panel_target=panel_white_noise,
            series_regime=series_regime,
            candidates=[dow_candidate],
        )
        cr = report.candidates["day_of_week"]
        assert cr.validated is False
        assert "day_of_week" in report.rejected_candidates

    def test_unstable_dow_fails(self, panel_unstable_dow, series_regime, dow_candidate):
        report = run_seasonality_analysis(
            panel_target=panel_unstable_dow,
            series_regime=series_regime,
            candidates=[dow_candidate],
        )
        cr = report.candidates["day_of_week"]
        # Should fail stability or consistency
        assert cr.validated is False

    def test_weak_consistency_fails(self, panel_weak_consistency, series_regime, dow_candidate):
        report = run_seasonality_analysis(
            panel_target=panel_weak_consistency,
            series_regime=series_regime,
            candidates=[dow_candidate],
        )
        cr = report.candidates["day_of_week"]
        assert cr.validated is False

    def test_no_candidates(self, panel_white_noise, series_regime):
        report = run_seasonality_analysis(
            panel_target=panel_white_noise,
            series_regime=series_regime,
            candidates=[],
        )
        assert len(report.candidates) == 0
        assert "No seasonal" in report.recommendation

    def test_periodogram_present(self, panel_white_noise, series_regime, dow_candidate):
        report = run_seasonality_analysis(
            panel_target=panel_white_noise,
            series_regime=series_regime,
            candidates=[dow_candidate],
        )
        assert "dominant_periods" in report.periodogram
