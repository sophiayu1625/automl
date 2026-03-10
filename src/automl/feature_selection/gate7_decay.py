"""Gate 7: IC decay analysis across forward-looking lags."""

import numpy as np
import pandas as pd

from automl.feature_selection.gate4_ic import (
    compute_cross_sectional_ic,
    compute_ic_stats,
)

_DEFAULT_LAGS = [1, 2, 5, 10, 21]


def compute_ic_decay(
    panel_target: pd.DataFrame,
    panel_feature: pd.DataFrame,
    decay_lags: list[int] | None = None,
    newey_west_lags: int = 1,
) -> dict:
    """Measure IC decay by shifting target forward.

    Parameters
    ----------
    panel_target, panel_feature : pd.DataFrame
        Rows=timestamps, cols=series_id.
    decay_lags : list[int]
        Lags to evaluate (in downsampled units).
    newey_west_lags : int
        Lags for Newey-West SE.

    Returns
    -------
    dict with decay_curve, tstat_curve, half_life, decay_classification.
    """
    if decay_lags is None:
        decay_lags = _DEFAULT_LAGS

    decay_curve: dict[int, float] = {}
    tstat_curve: dict[int, float] = {}

    for lag in decay_lags:
        target_shifted = panel_target.shift(-lag)
        ic = compute_cross_sectional_ic(target_shifted, panel_feature)
        stats = compute_ic_stats(ic, newey_west_lags)
        decay_curve[lag] = stats["ic_mean"]
        tstat_curve[lag] = stats["t_stat_nw"]

    # Half-life estimation
    half_life = _estimate_half_life(decay_curve, decay_lags)

    if np.isnan(half_life):
        classification = "medium"
    elif half_life <= 2:
        classification = "fast"
    elif half_life <= 7:
        classification = "medium"
    else:
        classification = "slow"

    return {
        "decay_curve": decay_curve,
        "tstat_curve": tstat_curve,
        "half_life": half_life,
        "decay_classification": classification,
    }


def _estimate_half_life(decay_curve: dict[int, float], lags: list[int]) -> float:
    """Interpolate half-life: lag where IC = 50% of lag-1 value."""
    if not lags or lags[0] not in decay_curve:
        return np.nan

    ic_at_1 = decay_curve[lags[0]]
    if np.isnan(ic_at_1) or ic_at_1 == 0:
        return np.nan

    target_ic = abs(ic_at_1) * 0.5

    for i in range(len(lags) - 1):
        ic_a = abs(decay_curve.get(lags[i], np.nan))
        ic_b = abs(decay_curve.get(lags[i + 1], np.nan))
        if np.isnan(ic_a) or np.isnan(ic_b):
            continue
        if ic_a >= target_ic >= ic_b and ic_a != ic_b:
            # Linear interpolation
            frac = (ic_a - target_ic) / (ic_a - ic_b)
            return float(lags[i] + frac * (lags[i + 1] - lags[i]))

    # IC never drops to half
    return float(lags[-1])
