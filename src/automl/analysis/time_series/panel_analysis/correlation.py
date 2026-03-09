"""6a: Correlation matrix with regime-based summaries."""

import numpy as np
import pandas as pd


def compute_correlation_matrix(
    panel_target: pd.DataFrame,
    series_regime: pd.Series,
    min_overlap: int = 100,
) -> dict:
    """Compute pairwise Pearson correlations with within/cross-regime summaries.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows = timestamps, cols = series_id.
    series_regime : pd.Series
        Index = series_id, values = structural group label.
    min_overlap : int
        Minimum overlapping non-NaN observations for a valid pair.

    Returns
    -------
    dict with keys: corr_matrix, within_regime_corr, cross_regime_corr.
    """
    series_ids = panel_target.columns.tolist()
    n = len(series_ids)
    corr_mat = pd.DataFrame(np.nan, index=series_ids, columns=series_ids)

    for i in range(n):
        corr_mat.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            si, sj = series_ids[i], series_ids[j]
            pair = panel_target[[si, sj]].dropna()
            if len(pair) >= min_overlap:
                c = pair[si].corr(pair[sj])
                corr_mat.loc[si, sj] = c
                corr_mat.loc[sj, si] = c

    # Within-regime and cross-regime averages
    within = {}
    cross = {}
    regimes = series_regime.reindex(series_ids).dropna()
    unique_regimes = sorted(regimes.unique())

    for r in unique_regimes:
        members = regimes[regimes == r].index.tolist()
        vals = []
        for i, si in enumerate(members):
            for sj in members[i + 1:]:
                v = corr_mat.loc[si, sj]
                if not np.isnan(v):
                    vals.append(v)
        within[str(r)] = {
            "mean": float(np.mean(vals)) if vals else np.nan,
            "median": float(np.median(vals)) if vals else np.nan,
            "n_pairs": len(vals),
        }

    for i, r1 in enumerate(unique_regimes):
        m1 = regimes[regimes == r1].index.tolist()
        for r2 in unique_regimes[i + 1:]:
            m2 = regimes[regimes == r2].index.tolist()
            vals = []
            for si in m1:
                for sj in m2:
                    v = corr_mat.loc[si, sj]
                    if not np.isnan(v):
                        vals.append(v)
            key = f"{r1}_vs_{r2}"
            cross[key] = {
                "mean": float(np.mean(vals)) if vals else np.nan,
                "median": float(np.median(vals)) if vals else np.nan,
                "n_pairs": len(vals),
            }

    return {
        "corr_matrix": corr_mat,
        "within_regime_corr": within,
        "cross_regime_corr": cross,
    }
