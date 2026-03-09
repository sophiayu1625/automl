"""Step 3: Sub-sample stability and cross-series consistency."""

import numpy as np
import pandas as pd

from automl.analysis.time_series.seasonality_analysis.candidates import SeasonalCandidate
from automl.analysis.time_series.seasonality_analysis.step2_tests import (
    check_categorical_seasonality,
    check_continuous_seasonality,
)

_STABILITY_THRESHOLD = 0.67
_DIRECTION_THRESHOLD = 0.75
_MIN_SERIES_FOR_CONSISTENCY = 5


def _run_test(series, grouping_var, candidate_type, significance_level):
    if candidate_type == "categorical":
        return check_categorical_seasonality(series, grouping_var, significance_level)
    else:
        return check_continuous_seasonality(series, grouping_var, significance_level)


def check_stability(
    series: pd.Series,
    grouping_var: pd.Series,
    candidate: SeasonalCandidate,
    n_splits: int = 3,
    significance_level: float = 0.05,
) -> dict:
    """Test temporal stability by splitting into sub-periods.

    Returns
    -------
    dict with n_significant, direction_consistent, stable.
    """
    aligned = pd.DataFrame({"value": series, "group": grouping_var}).dropna()
    n = len(aligned)
    split_size = n // n_splits

    if split_size < 20:
        return {"n_significant": 0, "direction_consistent": False, "stable": False}

    significant_count = 0
    effects = []

    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i < n_splits - 1 else n
        sub = aligned.iloc[start:end]

        result = _run_test(
            sub["value"], sub["group"],
            candidate.candidate_type, significance_level,
        )

        if result.get("reject", False):
            significant_count += 1

        # Track direction: for categorical use max-min of group means,
        # for continuous use coefficient sign
        if candidate.candidate_type == "categorical":
            gm = result.get("group_means", {})
            if gm:
                vals = list(gm.values())
                effects.append(max(vals) - min(vals))
            else:
                effects.append(0.0)
        else:
            effects.append(result.get("coef", 0.0))

    # Direction consistency: all effects have the same sign
    nonzero = [e for e in effects if e != 0]
    if nonzero:
        signs = [np.sign(e) for e in nonzero]
        direction_consistent = all(s == signs[0] for s in signs)
    else:
        direction_consistent = False

    frac_significant = significant_count / n_splits
    stable = frac_significant >= _STABILITY_THRESHOLD and direction_consistent

    return {
        "n_significant": significant_count,
        "direction_consistent": direction_consistent,
        "stable": stable,
    }


def compute_cross_series_consistency(
    per_series_results: dict[str, dict],
    significance_level: float = 0.05,
) -> dict:
    """Measure fraction of series showing significant effects in same direction.

    Parameters
    ----------
    per_series_results : dict
        Keyed by series_id; each value is a test result dict with
        'reject', 'group_means' or 'coef'.

    Returns
    -------
    dict with pct_significant, pct_same_direction, consistent.
    """
    if len(per_series_results) < _MIN_SERIES_FOR_CONSISTENCY:
        return {
            "pct_significant": np.nan,
            "pct_same_direction": np.nan,
            "consistent": False,
        }

    n_total = len(per_series_results)
    n_sig = 0
    directions = []

    for sid, res in per_series_results.items():
        if res.get("reject", False):
            n_sig += 1
            # Direction
            if "coef" in res:
                directions.append(np.sign(res["coef"]))
            elif "group_means" in res and res["group_means"]:
                vals = list(res["group_means"].values())
                directions.append(np.sign(max(vals) - min(vals)))

    pct_sig = n_sig / n_total
    if directions:
        most_common = max(set(directions), key=directions.count)
        pct_same = directions.count(most_common) / len(directions)
    else:
        pct_same = 0.0

    consistent = (
        pct_sig >= 0.6
        and pct_same >= _DIRECTION_THRESHOLD
    )

    return {
        "pct_significant": float(pct_sig),
        "pct_same_direction": float(pct_same),
        "consistent": consistent,
    }
