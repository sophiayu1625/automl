"""6c: Rolling mean pairwise correlation within regimes."""

import numpy as np
import pandas as pd


def compute_rolling_correlation(
    panel_target: pd.DataFrame,
    series_regime: pd.Series,
    rolling_window: str = "63D",
    min_periods: int = 20,
) -> pd.DataFrame:
    """Track time-varying mean pairwise correlations within each regime.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows = timestamps (DatetimeIndex), cols = series_id.
    series_regime : pd.Series
        Index = series_id, values = structural group label.
    rolling_window : str
        Window size for rolling correlation (e.g. '63D').
    min_periods : int
        Minimum observations in a window.

    Returns
    -------
    pd.DataFrame — index = timestamps, columns = regime labels,
                   values = mean pairwise correlation.
    """
    regimes = series_regime.reindex(panel_target.columns).dropna()
    unique_regimes = sorted(regimes.unique())
    result_cols: dict[str, pd.Series] = {}

    for r in unique_regimes:
        members = regimes[regimes == r].index.tolist()
        if len(members) < 2:
            continue

        sub = panel_target[members]
        rolling_corrs = []

        # Compute rolling correlation for each pair, then average
        for i, si in enumerate(members):
            for sj in members[i + 1:]:
                rc = (
                    sub[si]
                    .rolling(rolling_window, min_periods=min_periods)
                    .corr(sub[sj])
                )
                rolling_corrs.append(rc)

        if rolling_corrs:
            stacked = pd.concat(rolling_corrs, axis=1)
            result_cols[str(r)] = stacked.mean(axis=1)

    if not result_cols:
        return pd.DataFrame(index=panel_target.index)

    return pd.DataFrame(result_cols, index=panel_target.index)
