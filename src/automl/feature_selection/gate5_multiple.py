"""Gate 5: Multiple testing correction (Benjamini-Hochberg FDR)."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def apply_gate5(
    ic_stats_per_feature: dict[str, dict],
    fdr_q: float = 0.10,
    ic_mean_threshold: float = 0.02,
    icir_threshold: float = 0.5,
    tstat_threshold: float = 2.0,
) -> dict[str, dict]:
    """Apply BH FDR correction and combined threshold.

    Parameters
    ----------
    ic_stats_per_feature : dict
        Keyed by feature name; each value has 'p_value_nw', 'ic_mean',
        'icir', 't_stat_nw'.
    fdr_q : float
        FDR significance level.
    ic_mean_threshold, icir_threshold, tstat_threshold : float
        Thresholds for combined pass.

    Returns
    -------
    dict — {feature_name: {bh_passed, combined_passed, gate5_passed, bh_adjusted_pvalue}}.
    """
    feature_names = sorted(ic_stats_per_feature.keys())
    pvalues = []
    for f in feature_names:
        p = ic_stats_per_feature[f].get("p_value_nw", np.nan)
        pvalues.append(p if not np.isnan(p) else 1.0)

    # BH correction
    if len(pvalues) > 0:
        reject, adj_pvals, _, _ = multipletests(pvalues, alpha=fdr_q, method="fdr_bh")
    else:
        reject, adj_pvals = np.array([]), np.array([])

    results: dict = {}
    for i, f in enumerate(feature_names):
        stats = ic_stats_per_feature[f]
        bh_passed = bool(reject[i])
        bh_adj = float(adj_pvals[i])

        # Combined threshold
        ic_mean = stats.get("ic_mean", np.nan)
        icir = stats.get("icir", np.nan)
        tstat = stats.get("t_stat_nw", np.nan)

        combined = (
            _safe_gt(ic_mean, ic_mean_threshold)
            and _safe_gt(icir, icir_threshold)
            and _safe_gt(tstat, tstat_threshold)
        )

        results[f] = {
            "bh_passed": bh_passed,
            "combined_passed": combined,
            "gate5_passed": bh_passed or combined,
            "bh_adjusted_pvalue": bh_adj,
        }

    return results


def _safe_gt(val, threshold) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    return val > threshold
