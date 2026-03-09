"""SeasonalCandidate dataclass and built-in candidate constructors."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SeasonalCandidate:
    """Definition of a testable seasonal pattern."""

    name: str
    grouping_var: pd.Series
    candidate_type: str  # 'categorical' | 'continuous'
    seasonal_period: Optional[int] = None
    description: str = ""


def make_day_of_week_candidate(timestamps: pd.DatetimeIndex) -> SeasonalCandidate:
    """Day-of-week grouping (0=Mon … 6=Sun)."""
    grouping = pd.Series(timestamps.dayofweek, index=timestamps, name="day_of_week")
    return SeasonalCandidate(
        name="day_of_week",
        grouping_var=grouping,
        candidate_type="categorical",
        seasonal_period=5,
        description="Day-of-week effect (trading days)",
    )


def make_hour_of_day_candidate(timestamps: pd.DatetimeIndex) -> SeasonalCandidate:
    """Hour-of-day grouping."""
    grouping = pd.Series(timestamps.hour, index=timestamps, name="hour_of_day")
    return SeasonalCandidate(
        name="hour_of_day",
        grouping_var=grouping,
        candidate_type="categorical",
        seasonal_period=24,
        description="Hour-of-day effect",
    )


def make_month_end_candidate(
    timestamps: pd.DatetimeIndex, n_days: int = 3,
) -> SeasonalCandidate:
    """Binary: within *n_days* of month end."""
    days_to_end = pd.Series(
        (timestamps + pd.offsets.MonthEnd(0) - timestamps).days,
        index=timestamps,
        name="month_end",
    )
    grouping = (days_to_end <= n_days).astype(int)
    grouping.name = "month_end"
    return SeasonalCandidate(
        name="month_end",
        grouping_var=grouping,
        candidate_type="categorical",
        description=f"Within {n_days} days of month-end",
    )


def make_quarter_end_candidate(
    timestamps: pd.DatetimeIndex, n_days: int = 5,
) -> SeasonalCandidate:
    """Binary: within *n_days* of quarter end."""
    days_to_end = pd.Series(
        (timestamps + pd.offsets.QuarterEnd(0) - timestamps).days,
        index=timestamps,
        name="quarter_end",
    )
    grouping = (days_to_end <= n_days).astype(int)
    grouping.name = "quarter_end"
    return SeasonalCandidate(
        name="quarter_end",
        grouping_var=grouping,
        candidate_type="categorical",
        description=f"Within {n_days} days of quarter-end",
    )
