"""4c: Stationarity tests on downsampled level series."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

_MIN_OBS = 50


def _hurst_rs(series: np.ndarray) -> float:
    """Estimate Hurst exponent via rescaled-range (R/S) analysis."""
    n = len(series)
    if n < 20:
        return np.nan

    max_k = n // 2
    sizes = []
    rs_vals = []

    for k in [int(2 ** i) for i in range(2, int(np.log2(max_k)) + 1)]:
        if k > max_k:
            break
        n_blocks = n // k
        rs_block = []
        for b in range(n_blocks):
            block = series[b * k : (b + 1) * k]
            mean_b = np.mean(block)
            cumdev = np.cumsum(block - mean_b)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)
            if s > 0:
                rs_block.append(r / s)
        if rs_block:
            sizes.append(k)
            rs_vals.append(np.mean(rs_block))

    if len(sizes) < 2:
        return np.nan

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_vals)
    slope, _ = np.polyfit(log_sizes, log_rs, 1)
    return float(slope)


def _downsample(series: pd.Series, k: int) -> pd.Series:
    if k <= 1:
        return series
    return series.iloc[::k].copy()


def compute_stationarity(
    level: pd.Series,
    sampling_freq: int,
    horizon: int,
) -> dict:
    """Run ADF, KPSS, Zivot-Andrews, and Hurst exponent on downsampled level series.

    Parameters
    ----------
    level : pd.Series
        The level (price) series — NOT the differenced target.
    sampling_freq, horizon : int
        Used to compute downsample factor k = horizon // sampling_freq.

    Returns
    -------
    dict — flat dictionary with scalar values.
    """
    k = max(1, horizon // sampling_freq)
    ds = _downsample(level.dropna(), k)
    out: dict = {"stat_downsample_k": k, "stat_n": len(ds)}

    if len(ds) < _MIN_OBS:
        for key in [
            "adf_stat", "adf_pval",
            "kpss_stat", "kpss_pval",
            "za_stat", "za_pval", "za_breakpoint",
            "hurst",
        ]:
            out[key] = np.nan
        out["stat_error"] = True
        return out

    arr = ds.values

    # ADF
    try:
        adf_result = adfuller(arr, autolag="AIC")
        out["adf_stat"] = float(adf_result[0])
        out["adf_pval"] = float(adf_result[1])
    except Exception:
        out["adf_stat"] = np.nan
        out["adf_pval"] = np.nan

    # KPSS
    try:
        kpss_stat, kpss_pval, _, _ = kpss(arr, regression="c", nlags="auto")
        out["kpss_stat"] = float(kpss_stat)
        out["kpss_pval"] = float(kpss_pval)
    except Exception:
        out["kpss_stat"] = np.nan
        out["kpss_pval"] = np.nan

    # Zivot-Andrews
    try:
        za_result = zivot_andrews(arr, trim=0.15)
        out["za_stat"] = float(za_result[0])
        out["za_pval"] = float(za_result[1])
        out["za_breakpoint"] = int(za_result[2])
    except Exception:
        out["za_stat"] = np.nan
        out["za_pval"] = np.nan
        out["za_breakpoint"] = np.nan

    # Hurst exponent
    try:
        out["hurst"] = _hurst_rs(arr)
    except Exception:
        out["hurst"] = np.nan

    out["stat_error"] = False
    return out
