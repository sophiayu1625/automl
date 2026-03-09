"""Step 2: Formal statistical tests — F-test, regression, ACF, periodogram."""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

_MIN_GROUP_OBS = 5


def check_categorical_seasonality(
    series: pd.Series,
    grouping_var: pd.Series,
    significance_level: float = 0.05,
) -> dict:
    """One-way F-test for categorical seasonal pattern.

    Returns
    -------
    dict with f_stat, p_value, reject, r_squared, group_means.
    """
    df = pd.DataFrame({"value": series, "group": grouping_var}).dropna()
    groups_dict = {
        label: g["value"].values
        for label, g in df.groupby("group")
        if len(g) >= _MIN_GROUP_OBS
    }

    if len(groups_dict) < 2:
        return {
            "f_stat": np.nan, "p_value": np.nan, "reject": False,
            "r_squared": np.nan, "group_means": {},
        }

    groups_list = list(groups_dict.values())
    f_stat, p_value = f_oneway(*groups_list)

    # R-squared from ANOVA
    grand_mean = df["value"].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_list)
    ss_total = np.sum((df["value"].values - grand_mean) ** 2)
    r_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    group_means = {str(k): float(np.mean(v)) for k, v in groups_dict.items()}

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "reject": p_value < significance_level,
        "r_squared": r_sq,
        "group_means": group_means,
    }


def check_continuous_seasonality(
    series: pd.Series,
    grouping_var: pd.Series,
    significance_level: float = 0.05,
) -> dict:
    """OLS regression t-test for continuous predictor with Newey-West SE.

    Returns
    -------
    dict with coef, t_stat, p_value, reject, r_squared.
    """
    df = pd.DataFrame({"y": series, "x": grouping_var}).dropna()

    if len(df) < 10:
        return {
            "coef": np.nan, "t_stat": np.nan, "p_value": np.nan,
            "reject": False, "r_squared": np.nan,
        }

    X = add_constant(df["x"].values)
    model = OLS(df["y"].values, X)

    try:
        # Newey-West for serial correlation robustness
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": None})
    except Exception:
        res = model.fit()

    coef = float(res.params[1])
    t_stat = float(res.tvalues[1])
    p_value = float(res.pvalues[1])

    return {
        "coef": coef,
        "t_stat": t_stat,
        "p_value": p_value,
        "reject": p_value < significance_level,
        "r_squared": float(res.rsquared),
    }


def check_acf_at_seasonal_lag(
    series: pd.Series,
    seasonal_period: int,
    significance_level: float = 0.05,
) -> dict:
    """Check ACF at seasonal lag multiples.

    Returns
    -------
    dict with acf_at_period, acf_significant, acf_at_2period, acf_at_3period.
    """
    x = series.dropna().values
    max_lag = min(seasonal_period * 3, len(x) // 2)

    if max_lag < seasonal_period or len(x) < 50:
        return {
            "acf_at_period": np.nan,
            "acf_significant": None,
            "acf_at_2period": np.nan,
            "acf_at_3period": np.nan,
        }

    acf_vals, confint = acf(x, nlags=max_lag, fft=True, alpha=significance_level)

    def _significant(lag: int) -> bool:
        if lag >= len(acf_vals):
            return False
        lower, upper = confint[lag]
        # CI excludes zero → significant
        return bool(lower > 0 or upper < 0)

    p1 = seasonal_period
    p2 = seasonal_period * 2 if seasonal_period * 2 <= max_lag else None
    p3 = seasonal_period * 3 if seasonal_period * 3 <= max_lag else None

    return {
        "acf_at_period": float(acf_vals[p1]),
        "acf_significant": _significant(p1),
        "acf_at_2period": float(acf_vals[p2]) if p2 is not None else np.nan,
        "acf_at_3period": float(acf_vals[p3]) if p3 is not None else np.nan,
    }


def compute_periodogram(
    panel_target: pd.DataFrame,
    sampling_freq: str = "1h",
    top_n_peaks: int = 5,
) -> dict:
    """Compute averaged periodogram across series and find dominant periods.

    Returns
    -------
    dict with dominant_periods, dominant_powers, fishers_g_stat, fishers_g_pvalue.
    """
    spectra = []
    for col in panel_target.columns:
        x = panel_target[col].dropna().values
        if len(x) < 50:
            continue
        x_centered = x - np.mean(x)
        fft_vals = np.fft.rfft(x_centered)
        power = np.abs(fft_vals[1:]) ** 2  # exclude DC
        spectra.append(power / len(x))

    if not spectra:
        return {
            "dominant_periods": [],
            "dominant_powers": [],
            "fishers_g_stat": np.nan,
            "fishers_g_pvalue": np.nan,
        }

    # Average across series (truncate to shortest)
    min_len = min(len(s) for s in spectra)
    avg_power = np.mean([s[:min_len] for s in spectra], axis=0)

    # Dominant periods
    n_fft = (min_len) * 2  # approx original length
    freqs = np.arange(1, min_len + 1) / n_fft
    periods = 1.0 / freqs

    top_idx = np.argsort(avg_power)[::-1][:top_n_peaks]
    dominant_periods = periods[top_idx].tolist()
    dominant_powers = avg_power[top_idx].tolist()

    # Fisher's g-test (max power / sum of powers)
    g_stat = float(np.max(avg_power) / np.sum(avg_power))
    # Approximate p-value: P(G > g) ≈ n * (1-g)^(n-1) for large n
    n_freq = len(avg_power)
    g_pvalue = float(min(1.0, n_freq * (1 - g_stat) ** (n_freq - 1)))

    return {
        "dominant_periods": dominant_periods,
        "dominant_powers": dominant_powers,
        "fishers_g_stat": g_stat,
        "fishers_g_pvalue": g_pvalue,
    }
