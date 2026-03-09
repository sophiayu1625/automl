"""Tests for candidates.py — SeasonalCandidate constructors."""

import pandas as pd

from automl.analysis.time_series.seasonality_analysis.candidates import (
    make_day_of_week_candidate,
    make_hour_of_day_candidate,
    make_month_end_candidate,
    make_quarter_end_candidate,
)


class TestCandidateConstructors:
    def test_day_of_week(self, timestamps):
        cand = make_day_of_week_candidate(timestamps)
        assert cand.name == "day_of_week"
        assert cand.candidate_type == "categorical"
        assert cand.seasonal_period == 5
        assert len(cand.grouping_var) == len(timestamps)
        assert set(cand.grouping_var.unique()) <= set(range(7))

    def test_hour_of_day(self, timestamps):
        cand = make_hour_of_day_candidate(timestamps)
        assert cand.name == "hour_of_day"
        assert cand.seasonal_period == 24
        assert set(cand.grouping_var.unique()) == set(range(24))

    def test_month_end(self, timestamps):
        cand = make_month_end_candidate(timestamps, n_days=3)
        assert cand.name == "month_end"
        assert set(cand.grouping_var.unique()) <= {0, 1}

    def test_quarter_end(self, timestamps):
        cand = make_quarter_end_candidate(timestamps, n_days=5)
        assert cand.name == "quarter_end"
        assert set(cand.grouping_var.unique()) <= {0, 1}
