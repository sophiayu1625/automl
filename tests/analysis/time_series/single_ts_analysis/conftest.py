"""Shared fixtures for single_ts_analysis tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def sample_target(rng):
    """200-point target series with slight autocorrelation."""
    noise = rng.normal(0, 1, 200)
    ar = np.zeros(200)
    ar[0] = noise[0]
    for i in range(1, 200):
        ar[i] = 0.3 * ar[i - 1] + noise[i]
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    return pd.Series(ar, index=idx, name="target")


@pytest.fixture()
def sample_level(sample_target):
    """Cumulative sum of target — a non-stationary level series."""
    return sample_target.cumsum().rename("level")


@pytest.fixture()
def sample_regimes(rng):
    """Binary regime labels (0 / 1)."""
    labels = rng.choice([0, 1], size=200, p=[0.6, 0.4])
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    return pd.Series(labels, index=idx, name="regime")


@pytest.fixture()
def regime_names():
    return {0: "low_vol", 1: "high_vol"}


@pytest.fixture()
def short_series():
    """Series with < 50 observations — triggers graceful degradation."""
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    return pd.Series(np.arange(10, dtype=float), index=idx)
