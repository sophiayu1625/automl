"""Gate 2: Data quality checks and stationarity testing."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

_COVERAGE_THRESHOLD = 0.8
_ADF_REJECT_THRESHOLD = 0.7
_WINSOR_LOWER = 1
_WINSOR_UPPER = 99


def _hurst_rs(series: np.ndarray) -> float:
    """Estimate Hurst exponent via rescaled-range analysis."""
    n = len(series)
    if n < 20:
        return np.nan
    max_k = n // 2
    sizes, rs_vals = [], []
    for k in [int(2 ** i) for i in range(2, int(np.log2(max_k)) + 1)]:
        if k > max_k:
            break
        n_blocks = n // k
        rs_block = []
        for b in range(n_blocks):
            block = series[b * k : (b + 1) * k]
            cumdev = np.cumsum(block - np.mean(block))
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)
            if s > 0:
                rs_block.append(r / s)
        if rs_block:
            sizes.append(k)
            rs_vals.append(np.mean(rs_block))
    if len(sizes) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(sizes), np.log(rs_vals), 1)
    return float(slope)


def _winsorise(panel: pd.DataFrame) -> pd.DataFrame:
    """Winsorise each column at 1st/99th percentile."""
    out = panel.copy()
    for col in out.columns:
        lo = np.nanpercentile(out[col].values, _WINSOR_LOWER)
        hi = np.nanpercentile(out[col].values, _WINSOR_UPPER)
        out[col] = out[col].clip(lo, hi)
    return out


def apply_gate2(
    panel_features: dict[str, pd.DataFrame],
    significance_level: float = 0.05,
) -> dict:
    """Run data quality and stationarity checks per feature.

    Parameters
    ----------
    panel_features : dict[str, pd.DataFrame]
        Keyed by feature name; each DataFrame has rows=timestamps, cols=series_id.
    significance_level : float
        Alpha for ADF/KPSS tests.

    Returns
    -------
    dict with keys:
        'results': {feature_name: quality_dict},
        'winsorised_panels': {feature_name: pd.DataFrame}.
    """
    results: dict = {}
    winsorised: dict = {}

    for fname, panel in panel_features.items():
        total_cells = panel.size
        non_nan = panel.count().sum()
        coverage = float(non_nan / total_cells) if total_cells > 0 else 0.0

        # Outlier fraction
        vals = panel.values.flatten()
        vals_clean = vals[~np.isnan(vals)]
        if len(vals_clean) > 0:
            lo = np.percentile(vals_clean, _WINSOR_LOWER)
            hi = np.percentile(vals_clean, _WINSOR_UPPER)
            outlier_frac = float(np.mean((vals_clean < lo) | (vals_clean > hi)))
        else:
            outlier_frac = np.nan

        # Stationarity per series
        adf_rejects = 0
        kpss_rejects = 0
        hursts = []
        n_tested = 0

        for col in panel.columns:
            x = panel[col].dropna().values
            if len(x) < 50:
                continue
            n_tested += 1

            # ADF
            try:
                _, pval, *_ = adfuller(x, autolag="AIC")
                if pval < significance_level:
                    adf_rejects += 1
            except Exception:
                n_tested -= 1
                continue

            # KPSS
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, pval, *_ = kpss(x, regression="c", nlags="auto")
                if pval < significance_level:
                    kpss_rejects += 1
            except Exception:
                pass

            # Hurst
            try:
                hursts.append(_hurst_rs(x))
            except Exception:
                pass

        adf_reject_frac = adf_rejects / n_tested if n_tested > 0 else 0.0
        kpss_reject_frac = kpss_rejects / n_tested if n_tested > 0 else 0.0
        hurst_mean = float(np.nanmean(hursts)) if hursts else np.nan

        passed = coverage >= _COVERAGE_THRESHOLD and adf_reject_frac >= _ADF_REJECT_THRESHOLD

        results[fname] = {
            "coverage": coverage,
            "outlier_frac": outlier_frac,
            "adf_reject_frac": float(adf_reject_frac),
            "kpss_reject_frac": float(kpss_reject_frac),
            "hurst_mean": hurst_mean,
            "passed": passed,
        }

        winsorised[fname] = _winsorise(panel)

    return {"results": results, "winsorised_panels": winsorised}
