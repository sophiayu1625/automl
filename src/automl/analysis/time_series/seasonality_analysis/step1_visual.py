"""Step 1: Grouped statistics for visual inspection."""

import numpy as np
import pandas as pd
from scipy.stats import sem


_MIN_GROUP_OBS = 5


def compute_grouped_stats(
    series: pd.Series,
    grouping_var: pd.Series,
) -> pd.DataFrame:
    """Compute grouped mean/std/CI for each level of *grouping_var*.

    Parameters
    ----------
    series : pd.Series
        Target values.
    grouping_var : pd.Series
        Grouping labels aligned to *series*.

    Returns
    -------
    pd.DataFrame with columns: group_label, mean, std, sem, n,
        ci_lower_95, ci_upper_95.
    """
    df = pd.DataFrame({"value": series, "group": grouping_var}).dropna()
    groups = df.groupby("group")["value"]

    rows = []
    for label, g in groups:
        vals = g.values
        if len(vals) < _MIN_GROUP_OBS:
            continue
        se = sem(vals)
        m = np.mean(vals)
        rows.append({
            "group_label": label,
            "mean": float(m),
            "std": float(np.std(vals, ddof=1)),
            "sem": float(se),
            "n": len(vals),
            "ci_lower_95": float(m - 1.96 * se),
            "ci_upper_95": float(m + 1.96 * se),
        })

    return pd.DataFrame(rows)
