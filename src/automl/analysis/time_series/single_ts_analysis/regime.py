"""4f: Regime-conditional statistics with Levene test."""

import numpy as np
import pandas as pd
from scipy.stats import levene

from automl.analysis.time_series.single_ts_analysis.descriptive import _stats_for_array


def compute_regime_stats(
    target: pd.Series,
    regime_labels: pd.Series | None = None,
    regime_names: dict[int, str] | None = None,
) -> dict:
    """Compute regime-conditional distributions and cross-regime variance test.

    Parameters
    ----------
    target : pd.Series
        Target return / change series.
    regime_labels : pd.Series, optional
        Integer regime labels aligned to *target*.
    regime_names : dict, optional
        Mapping from regime int -> human-readable name.

    Returns
    -------
    dict — flat dictionary with regime stats and Levene test results.
    """
    out: dict = {}

    if regime_labels is None:
        out["regime_count"] = 0
        out["levene_stat"] = np.nan
        out["levene_pval"] = np.nan
        return out

    aligned = pd.DataFrame({"target": target, "regime": regime_labels}).dropna()
    unique_regimes = sorted(aligned["regime"].unique())
    out["regime_count"] = len(unique_regimes)

    # Per-regime stats
    groups = []
    for rid in unique_regimes:
        name = regime_names.get(int(rid), str(int(rid))) if regime_names else str(int(rid))
        subset = aligned.loc[aligned["regime"] == rid, "target"].values
        out[f"regime_{name}_n"] = len(subset)
        out[f"regime_{name}_frac"] = len(subset) / len(aligned) if len(aligned) > 0 else np.nan
        out.update(_stats_for_array(subset, prefix=f"regime_{name}"))
        if len(subset) >= 2:
            groups.append(subset)

    # Levene test for equality of variance across regimes
    if len(groups) >= 2:
        try:
            stat, pval = levene(*groups)
            out["levene_stat"] = float(stat)
            out["levene_pval"] = float(pval)
        except Exception:
            out["levene_stat"] = np.nan
            out["levene_pval"] = np.nan
    else:
        out["levene_stat"] = np.nan
        out["levene_pval"] = np.nan

    return out
