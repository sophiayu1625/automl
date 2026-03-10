"""Shared fixtures for feature_selection tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(77)


@pytest.fixture()
def timestamps():
    return pd.date_range("2020-01-01", periods=500, freq="h")


@pytest.fixture()
def series_ids():
    return [f"s{i}" for i in range(20)]


@pytest.fixture()
def panel_target(rng, timestamps, series_ids):
    data = rng.normal(0, 1, (len(timestamps), len(series_ids)))
    return pd.DataFrame(data, index=timestamps, columns=series_ids)


@pytest.fixture()
def series_regime(series_ids):
    labels = ["A"] * 10 + ["B"] * 10
    return pd.Series(labels, index=series_ids)


@pytest.fixture()
def macro_regime(timestamps):
    labels = np.where(np.arange(len(timestamps)) < 250, "calm", "stress")
    return pd.Series(labels, index=timestamps)


@pytest.fixture()
def strong_feature(rng, panel_target, timestamps, series_ids):
    """Feature with genuine IC — correlated with target."""
    noise = rng.normal(0, 0.3, panel_target.shape)
    return pd.DataFrame(
        panel_target.values + noise, index=timestamps, columns=series_ids,
    )


@pytest.fixture()
def noise_feature(rng, timestamps, series_ids):
    """Pure noise — no relationship with target."""
    return pd.DataFrame(
        rng.normal(0, 1, (len(timestamps), len(series_ids))),
        index=timestamps, columns=series_ids,
    )


@pytest.fixture()
def nonstationary_feature(rng, timestamps, series_ids):
    """Random walk — should fail Gate 2."""
    data = np.cumsum(rng.normal(0, 1, (len(timestamps), len(series_ids))), axis=0)
    return pd.DataFrame(data, index=timestamps, columns=series_ids)


@pytest.fixture()
def redundant_pair(strong_feature, rng, timestamps, series_ids):
    """Two features with |ρ| > 0.9."""
    f1 = strong_feature
    f2 = f1 + rng.normal(0, 0.05, f1.shape)
    f2 = pd.DataFrame(f2.values, index=timestamps, columns=series_ids)
    return f1, f2


@pytest.fixture()
def regime_flip_feature(rng, panel_target, timestamps, series_ids, series_regime):
    """Feature with sign flip across series regimes."""
    feat = pd.DataFrame(0.0, index=timestamps, columns=series_ids)
    for sid in series_ids:
        if series_regime[sid] == "A":
            feat[sid] = panel_target[sid] + rng.normal(0, 0.3, len(timestamps))
        else:
            feat[sid] = -panel_target[sid] + rng.normal(0, 0.3, len(timestamps))
    return feat
