"""Shared fixtures for seasonality_analysis tests."""

import numpy as np
import pandas as pd
import pytest

from automl.analysis.time_series.seasonality_analysis.candidates import (
    SeasonalCandidate,
    make_day_of_week_candidate,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def timestamps():
    """~90 days of hourly data."""
    return pd.date_range("2020-01-01", periods=2160, freq="h")


@pytest.fixture()
def panel_ids():
    return [f"s{i}" for i in range(10)]


@pytest.fixture()
def series_regime(panel_ids):
    return pd.Series(["A"] * 5 + ["B"] * 5, index=panel_ids)


@pytest.fixture()
def panel_white_noise(rng, timestamps, panel_ids):
    """No seasonality — pure white noise."""
    data = rng.normal(0, 1, (len(timestamps), len(panel_ids)))
    return pd.DataFrame(data, index=timestamps, columns=panel_ids)


@pytest.fixture()
def panel_dow_effect(rng, timestamps, panel_ids):
    """Monday +2 effect, other days 0, plus noise."""
    data = np.zeros((len(timestamps), len(panel_ids)))
    for j in range(len(panel_ids)):
        noise = rng.normal(0, 0.5, len(timestamps))
        effect = np.where(timestamps.dayofweek == 0, 2.0, 0.0)
        data[:, j] = effect + noise
    return pd.DataFrame(data, index=timestamps, columns=panel_ids)


@pytest.fixture()
def panel_unstable_dow(rng, timestamps, panel_ids):
    """DOW effect only in first half — should fail stability gate."""
    n = len(timestamps)
    half = n // 2
    data = np.zeros((n, len(panel_ids)))
    for j in range(len(panel_ids)):
        noise = rng.normal(0, 0.5, n)
        effect = np.where(timestamps.dayofweek == 0, 2.0, 0.0)
        effect[half:] = 0.0  # remove effect in second half
        data[:, j] = effect + noise
    return pd.DataFrame(data, index=timestamps, columns=panel_ids)


@pytest.fixture()
def panel_weak_consistency(rng, timestamps, panel_ids):
    """Only 30% of series have DOW effect — should fail consistency gate."""
    data = np.zeros((len(timestamps), len(panel_ids)))
    for j in range(len(panel_ids)):
        noise = rng.normal(0, 0.5, len(timestamps))
        if j < 3:  # only 3 out of 10 series
            effect = np.where(timestamps.dayofweek == 0, 2.0, 0.0)
        else:
            effect = np.zeros(len(timestamps))
        data[:, j] = effect + noise
    return pd.DataFrame(data, index=timestamps, columns=panel_ids)


@pytest.fixture()
def dow_candidate(timestamps):
    return make_day_of_week_candidate(timestamps)
