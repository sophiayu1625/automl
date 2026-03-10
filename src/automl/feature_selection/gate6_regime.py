"""Gate 6: Regime-conditional IC analysis."""

import numpy as np
import pandas as pd

from automl.feature_selection.gate4_ic import (
    compute_cross_sectional_ic,
    compute_ic_stats,
)


def compute_regime_ic(
    panel_target: pd.DataFrame,
    panel_feature: pd.DataFrame,
    series_regime: pd.Series,
    macro_regime: pd.Series,
    newey_west_lags: int = 1,
) -> dict:
    """Compute IC within each regime subset.

    Parameters
    ----------
    panel_target, panel_feature : pd.DataFrame
        Rows=timestamps, cols=series_id.
    series_regime : pd.Series
        Index=series_id, values=structural group label.
    macro_regime : pd.Series
        Index=timestamp, values=time-varying regime label.
    newey_west_lags : int
        Lags for Newey-West SE.

    Returns
    -------
    dict with by_series_regime, by_macro_regime, sign_flip flags, regime_stable,
         classification.
    """
    by_series = {}
    series_regimes = series_regime.reindex(panel_target.columns).dropna()
    unique_sr = sorted(series_regimes.unique())

    for r in unique_sr:
        members = series_regimes[series_regimes == r].index.tolist()
        if len(members) < 5:
            by_series[str(r)] = {"ic_mean": np.nan}
            continue
        ic = compute_cross_sectional_ic(
            panel_target[members], panel_feature.reindex(columns=members),
        )
        by_series[str(r)] = compute_ic_stats(ic, newey_west_lags)

    # By macro regime (restrict rows)
    by_macro = {}
    macro_aligned = macro_regime.reindex(panel_target.index).dropna()
    unique_mr = sorted(macro_aligned.unique())

    for r in unique_mr:
        mask = macro_aligned == r
        ts_sub = panel_target.loc[mask]
        feat_sub = panel_feature.reindex(index=ts_sub.index)
        if len(ts_sub) < 10:
            by_macro[str(r)] = {"ic_mean": np.nan}
            continue
        ic = compute_cross_sectional_ic(ts_sub, feat_sub)
        by_macro[str(r)] = compute_ic_stats(ic, newey_west_lags)

    # Sign flips
    sr_means = [v.get("ic_mean", np.nan) for v in by_series.values()]
    sr_means_valid = [m for m in sr_means if not np.isnan(m)]
    sign_flip_sr = _has_sign_flip(sr_means_valid)

    mr_means = [v.get("ic_mean", np.nan) for v in by_macro.values()]
    mr_means_valid = [m for m in mr_means if not np.isnan(m)]
    sign_flip_mr = _has_sign_flip(mr_means_valid)

    # Classification
    if sign_flip_sr and sign_flip_mr:
        classification = "dangerous"
        stable = False
    elif sign_flip_sr:
        classification = "series_conditional"
        stable = False
    elif sign_flip_mr:
        classification = "macro_conditional"
        stable = True
    else:
        classification = "unconditional"
        stable = True

    return {
        "by_series_regime": by_series,
        "by_macro_regime": by_macro,
        "sign_flip_across_series_regime": sign_flip_sr,
        "sign_flip_across_macro_regime": sign_flip_mr,
        "regime_stable": stable,
        "classification": classification,
    }


def _has_sign_flip(means: list[float]) -> bool:
    """Check if there are both positive and negative means."""
    if len(means) < 2:
        return False
    has_pos = any(m > 0 for m in means)
    has_neg = any(m < 0 for m in means)
    return has_pos and has_neg
