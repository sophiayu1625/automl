"""Gate 4: Walk-forward Information Coefficient (IC) computation."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist


def compute_cross_sectional_ic(
    panel_target: pd.DataFrame,
    panel_feature: pd.DataFrame,
    min_series: int = 10,
) -> pd.Series:
    """Compute cross-sectional Spearman IC at each timestamp.

    At each timestamp t: IC_t = SpearmanCorr(feature[:, t], target[:, t]).

    Returns pd.Series indexed by timestamp.
    """
    common_cols = panel_target.columns.intersection(panel_feature.columns)
    common_idx = panel_target.index.intersection(panel_feature.index)

    ic_values = []
    for ts in common_idx:
        tgt = panel_target.loc[ts, common_cols]
        feat = panel_feature.loc[ts, common_cols]
        valid = ~(tgt.isna() | feat.isna())
        if valid.sum() < min_series:
            ic_values.append(np.nan)
            continue
        rho, _ = spearmanr(feat[valid], tgt[valid])
        ic_values.append(float(rho))

    return pd.Series(ic_values, index=common_idx, name="ic")


def compute_ic_stats(
    ic_series: pd.Series,
    newey_west_lags: int = 1,
) -> dict:
    """Compute IC summary statistics with Newey-West adjusted t-statistic.

    Parameters
    ----------
    ic_series : pd.Series
        Cross-sectional IC time series.
    newey_west_lags : int
        Number of lags for Newey-West SE.

    Returns
    -------
    dict with ic_mean, ic_std, icir, ic_positive_frac, t_stat_nw, p_value_nw.
    """
    ic = ic_series.dropna().values
    T = len(ic)

    if T < 5:
        return {
            "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
            "ic_positive_frac": np.nan, "t_stat_nw": np.nan, "p_value_nw": np.nan,
        }

    ic_mean = float(np.mean(ic))
    ic_std = float(np.std(ic, ddof=1))

    if ic_std == 0:
        return {
            "ic_mean": ic_mean, "ic_std": 0.0, "icir": np.nan,
            "ic_positive_frac": float(np.mean(ic > 0)),
            "t_stat_nw": np.nan, "p_value_nw": np.nan,
        }

    icir = ic_mean / ic_std

    # Newey-West variance
    demeaned = ic - ic_mean
    gamma_0 = float(np.mean(demeaned ** 2))
    nw_var = gamma_0
    L = newey_west_lags
    for k in range(1, L + 1):
        if k >= T:
            break
        gamma_k = float(np.mean(demeaned[k:] * demeaned[:-k]))
        w_k = 1 - k / (L + 1)  # Bartlett weight
        nw_var += 2 * w_k * gamma_k

    se_nw = np.sqrt(nw_var / T)
    if se_nw == 0:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = float(ic_mean / se_nw)
        p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=T - 1)))

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_frac": float(np.mean(ic > 0)),
        "t_stat_nw": t_stat,
        "p_value_nw": p_value,
    }


def apply_gate4_decision(
    ic_stats: dict,
    ic_mean_threshold: float = 0.02,
    icir_threshold: float = 0.5,
    tstat_threshold: float = 2.0,
    positive_frac_threshold: float = 0.55,
) -> dict:
    """Apply Gate 4 pass/fail decision.

    Returns dict with passed bool and individual checks.
    """
    ic_mean = ic_stats.get("ic_mean", np.nan)
    icir = ic_stats.get("icir", np.nan)
    t_stat = ic_stats.get("t_stat_nw", np.nan)
    pos_frac = ic_stats.get("ic_positive_frac", np.nan)

    checks = {
        "ic_mean_pass": _gt(ic_mean, ic_mean_threshold),
        "icir_pass": _gt(icir, icir_threshold),
        "tstat_pass": _gt(t_stat, tstat_threshold),
        "positive_frac_pass": _gt(pos_frac, positive_frac_threshold),
    }
    passed = all(checks.values())

    return {"passed": passed, **checks}


def _gt(val, threshold) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    return val > threshold
