"""6d: Cross-regime comparison with statistical tests."""

import numpy as np
import pandas as pd
from scipy.stats import levene, ks_2samp, ttest_ind, kruskal

_ALPHA = 0.05
_PHASE4_METRICS = ["hurst", "garch_persistence", "garch_gamma", "desc_kurt", "acf_lag1"]


def compare_regimes(
    panel_target: pd.DataFrame,
    series_regime: pd.Series,
    results_df: pd.DataFrame,
) -> dict:
    """Compare distributional properties across series regimes.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows = timestamps, cols = series_id.
    series_regime : pd.Series
        Index = series_id, values = structural group label.
    results_df : pd.DataFrame
        Phase 4 output with index = series_id.

    Returns
    -------
    dict with pairwise test results and architecture recommendation.
    """
    regimes = series_regime.reindex(panel_target.columns).dropna()
    unique_regimes = sorted(regimes.unique())
    out: dict = {"pairwise": {}, "kruskal_wallis": {}}

    # Pool observations per regime
    pooled: dict[str, np.ndarray] = {}
    for r in unique_regimes:
        members = regimes[regimes == r].index.tolist()
        vals = panel_target[members].values.flatten()
        pooled[str(r)] = vals[~np.isnan(vals)]

    # Pairwise tests
    for i, r1 in enumerate(unique_regimes):
        for r2 in unique_regimes[i + 1:]:
            key = f"{r1}_vs_{r2}"
            a, b = pooled[str(r1)], pooled[str(r2)]
            pair: dict = {}

            if len(a) < 10 or len(b) < 10:
                for k in ["levene_stat", "levene_pvalue", "ks_stat", "ks_pvalue",
                           "ttest_stat", "ttest_pvalue"]:
                    pair[k] = np.nan
                pair["levene_reject"] = np.nan
                pair["ks_reject"] = np.nan
                out["pairwise"][key] = pair
                continue

            # Levene
            try:
                stat, pval = levene(a, b)
                pair["levene_stat"] = float(stat)
                pair["levene_pvalue"] = float(pval)
                pair["levene_reject"] = pval < _ALPHA
            except Exception:
                pair["levene_stat"] = np.nan
                pair["levene_pvalue"] = np.nan
                pair["levene_reject"] = np.nan

            # KS
            try:
                stat, pval = ks_2samp(a, b)
                pair["ks_stat"] = float(stat)
                pair["ks_pvalue"] = float(pval)
                pair["ks_reject"] = pval < _ALPHA
            except Exception:
                pair["ks_stat"] = np.nan
                pair["ks_pvalue"] = np.nan
                pair["ks_reject"] = np.nan

            # Welch t-test
            try:
                stat, pval = ttest_ind(a, b, equal_var=False)
                pair["ttest_stat"] = float(stat)
                pair["ttest_pvalue"] = float(pval)
            except Exception:
                pair["ttest_stat"] = np.nan
                pair["ttest_pvalue"] = np.nan

            # Tail differences
            pair["tail_p01_diff"] = float(np.percentile(a, 1) - np.percentile(b, 1))
            pair["tail_p99_diff"] = float(np.percentile(a, 99) - np.percentile(b, 99))

            out["pairwise"][key] = pair

    # Kruskal-Wallis on Phase 4 metrics across regimes
    results_with_regime = results_df.copy()
    results_with_regime["_regime"] = series_regime.reindex(results_df.index)
    results_with_regime = results_with_regime.dropna(subset=["_regime"])

    for metric in _PHASE4_METRICS:
        if metric not in results_with_regime.columns:
            out["kruskal_wallis"][metric] = np.nan
            continue
        groups = []
        for r in unique_regimes:
            g = results_with_regime.loc[
                results_with_regime["_regime"] == r, metric
            ].dropna().values
            if len(g) >= 2:
                groups.append(g)
        if len(groups) >= 2:
            try:
                _, pval = kruskal(*groups)
                out["kruskal_wallis"][metric] = float(pval)
            except Exception:
                out["kruskal_wallis"][metric] = np.nan
        else:
            out["kruskal_wallis"][metric] = np.nan

    # Architecture recommendation
    out["architecture_recommendation"] = _recommend(out)

    return out


def _recommend(results: dict) -> str:
    """Determine architecture recommendation based on test results."""
    pairwise = results.get("pairwise", {})
    if not pairwise:
        return "investigate_further"

    n_reject = 0
    n_tests = 0
    for pair_results in pairwise.values():
        for key in ["levene_reject", "ks_reject"]:
            val = pair_results.get(key)
            if val is not np.nan and val is not None:
                n_tests += 1
                if val:
                    n_reject += 1

    if n_tests == 0:
        return "investigate_further"

    reject_rate = n_reject / n_tests
    if reject_rate > 0.7:
        return "separate_models"
    elif reject_rate < 0.3:
        return "single_model_with_regime_feature"
    else:
        return "investigate_further"
