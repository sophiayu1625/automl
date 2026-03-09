"""4b: Autocorrelation analysis on downsampled series."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

_ACF_LAGS = [1, 2, 5, 10, 20]
_MIN_OBS = 50


def _downsample(series: pd.Series, k: int) -> pd.Series:
    """Non-overlapping downsample: take every k-th observation."""
    if k <= 1:
        return series
    return series.iloc[::k].copy()


def compute_autocorrelation(
    target: pd.Series,
    sampling_freq: int,
    horizon: int,
) -> dict:
    """Compute ACF values and Ljung-Box test on downsampled target series.

    Parameters
    ----------
    target : pd.Series
        Target return / change series (full frequency).
    sampling_freq : int
        Sampling frequency in the same unit as *horizon* (e.g. hours).
    horizon : int
        Prediction horizon in the same unit.

    Returns
    -------
    dict  — flat dictionary with scalar values.
    """
    k = max(1, horizon // sampling_freq)
    ds = _downsample(target.dropna(), k)
    out: dict = {"acf_downsample_k": k, "acf_n": len(ds)}

    max_lag = max(_ACF_LAGS)

    if len(ds) < _MIN_OBS or len(ds) <= max_lag + 1:
        for lag in _ACF_LAGS:
            out[f"acf_lag{lag}"] = np.nan
        out["ljungbox_stat"] = np.nan
        out["ljungbox_pval"] = np.nan
        out["acf_error"] = True
        return out

    try:
        acf_vals = acf(ds, nlags=max_lag, fft=True)
        for lag in _ACF_LAGS:
            out[f"acf_lag{lag}"] = float(acf_vals[lag])

        lb = acorr_ljungbox(ds, lags=[max_lag], return_df=True)
        out["ljungbox_stat"] = float(lb["lb_stat"].iloc[0])
        out["ljungbox_pval"] = float(lb["lb_pvalue"].iloc[0])
        out["acf_error"] = False
    except Exception:
        for lag in _ACF_LAGS:
            out[f"acf_lag{lag}"] = np.nan
        out["ljungbox_stat"] = np.nan
        out["ljungbox_pval"] = np.nan
        out["acf_error"] = True

    return out
