"""Main entry point: analyse_series() orchestrates all sub-analyses."""

import pandas as pd

from automl.analysis.time_series.single_ts_analysis.descriptive import compute_descriptive_stats
from automl.analysis.time_series.single_ts_analysis.autocorrelation import compute_autocorrelation
from automl.analysis.time_series.single_ts_analysis.stationarity import compute_stationarity
from automl.analysis.time_series.single_ts_analysis.volatility import compute_garch, compute_realized_variance
from automl.analysis.time_series.single_ts_analysis.regime import compute_regime_stats


def analyse_series(
    series_id: str,
    target: pd.Series,
    level: pd.Series,
    regime_labels: pd.Series | None = None,
    sampling_freq: int = 1,
    horizon: int = 1,
    regime_names: dict[int, str] | None = None,
) -> dict:
    """Run full single time-series analysis and return a flat dictionary.

    Parameters
    ----------
    series_id : str
        Identifier for this series.
    target : pd.Series
        Target return / change series.
    level : pd.Series
        Level (price) series for stationarity tests.
    regime_labels : pd.Series, optional
        Integer regime labels aligned to *target*.
    sampling_freq : int
        Sampling frequency in common units (e.g. 1 for hourly).
    horizon : int
        Prediction horizon in the same units.
    regime_names : dict, optional
        Mapping from regime int -> human-readable name.

    Returns
    -------
    dict — flat dictionary with all scalar results, suitable for
           pd.DataFrame conversion.
    """
    result: dict = {"series_id": series_id}

    # 4a — descriptive statistics (overall + per-regime)
    result.update(compute_descriptive_stats(target, regime_labels, regime_names))

    # 4b — autocorrelation on downsampled series
    result.update(compute_autocorrelation(target, sampling_freq, horizon))

    # 4c — stationarity tests on downsampled level series
    result.update(compute_stationarity(level, sampling_freq, horizon))

    # 4d — GARCH volatility modeling
    result.update(compute_garch(target))

    # 4e — realized variance (full frequency)
    result.update(compute_realized_variance(target, sampling_freq, horizon))

    # 4f — regime-conditional analysis
    result.update(compute_regime_stats(target, regime_labels, regime_names))

    return result
