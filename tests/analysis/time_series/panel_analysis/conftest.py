"""Shared fixtures for panel_analysis tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def rng():
    return np.random.default_rng(123)


@pytest.fixture()
def panel_ids():
    return [f"s{i}" for i in range(10)]


@pytest.fixture()
def timestamps():
    return pd.date_range("2020-01-01", periods=500, freq="h")


@pytest.fixture()
def series_regime(panel_ids):
    """First 5 series in regime A, last 5 in regime B."""
    labels = ["A"] * 5 + ["B"] * 5
    return pd.Series(labels, index=panel_ids, name="regime")


@pytest.fixture()
def macro_regime(timestamps):
    """Alternating macro regimes every 250 timestamps."""
    labels = np.where(np.arange(len(timestamps)) < 250, "calm", "stress")
    return pd.Series(labels, index=timestamps, name="macro")


@pytest.fixture()
def panel_target_independent(rng, timestamps, panel_ids):
    """10 independent series — expect near-zero correlations."""
    data = rng.normal(0, 1, (len(timestamps), len(panel_ids)))
    return pd.DataFrame(data, index=timestamps, columns=panel_ids)


@pytest.fixture()
def panel_target_common_factor(rng, timestamps, panel_ids):
    """10 series driven by a common factor — expect high correlations."""
    factor = rng.normal(0, 1, len(timestamps))
    data = np.column_stack([
        factor + rng.normal(0, 0.1, len(timestamps)) for _ in panel_ids
    ])
    return pd.DataFrame(data, index=timestamps, columns=panel_ids)


@pytest.fixture()
def panel_level(panel_target_independent):
    """Cumulative sum to create level panel."""
    return panel_target_independent.cumsum()


@pytest.fixture()
def results_df(rng, panel_ids):
    """Fake Phase 4 results."""
    return pd.DataFrame({
        "hurst": rng.uniform(0.3, 0.8, len(panel_ids)),
        "garch_persistence": rng.uniform(0.5, 0.99, len(panel_ids)),
        "garch_gamma": rng.uniform(0, 0.3, len(panel_ids)),
        "desc_kurt": rng.uniform(-1, 5, len(panel_ids)),
        "acf_lag1": rng.uniform(-0.1, 0.5, len(panel_ids)),
        "desc_std": rng.uniform(0.5, 2.0, len(panel_ids)),
    }, index=panel_ids)


@pytest.fixture()
def series_metadata(rng, panel_ids):
    """Fake metadata with boundary_distance."""
    return pd.DataFrame({
        "maturity": rng.uniform(1, 30, len(panel_ids)),
        "boundary_distance": np.concatenate([
            rng.uniform(0, 0.3, 4),   # near boundary
            rng.uniform(1.0, 3.0, 6), # core
        ]),
    }, index=panel_ids)


@pytest.fixture()
def panel_target_heteroscedastic(rng, timestamps, panel_ids, series_regime):
    """Regime A has low vol, regime B has high vol."""
    data = pd.DataFrame(index=timestamps, columns=panel_ids, dtype=float)
    for sid in panel_ids:
        r = series_regime[sid]
        scale = 0.5 if r == "A" else 3.0
        data[sid] = rng.normal(0, scale, len(timestamps))
    return data
