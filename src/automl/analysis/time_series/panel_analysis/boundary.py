"""6e: Boundary effects analysis."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def analyse_boundary_effects(
    results_df: pd.DataFrame,
    series_metadata: pd.DataFrame,
    series_regime: pd.Series,
    boundary_threshold: float = 0.5,
) -> dict:
    """Test whether proximity to regime boundaries predicts statistical properties.

    Parameters
    ----------
    results_df : pd.DataFrame
        Phase 4 output with index = series_id.
    series_metadata : pd.DataFrame
        Index = series_id; must contain 'boundary_distance' column.
    series_regime : pd.Series
        Index = series_id, values = structural group label.
    boundary_threshold : float
        Series with boundary_distance <= this are classified as near-boundary.

    Returns
    -------
    dict with per-regime, per-metric boundary effect test results.
    """
    if "boundary_distance" not in series_metadata.columns:
        return {"error": "boundary_distance not in series_metadata"}

    merged = results_df.join(series_metadata[["boundary_distance"]], how="inner")
    merged["_regime"] = series_regime.reindex(merged.index)
    merged["_is_boundary"] = merged["boundary_distance"] <= boundary_threshold
    merged = merged.dropna(subset=["_regime"])

    unique_regimes = sorted(merged["_regime"].unique())
    test_metrics = [
        c for c in results_df.columns
        if results_df[c].dtype in [np.float64, np.float32, float]
        and not c.startswith("_")
    ]

    out: dict = {}

    for r in unique_regimes:
        sub = merged[merged["_regime"] == r]
        core = sub[~sub["_is_boundary"]]
        boundary = sub[sub["_is_boundary"]]
        regime_key = str(r)
        out[regime_key] = {}

        for metric in test_metrics:
            core_vals = core[metric].dropna().values
            bnd_vals = boundary[metric].dropna().values
            entry: dict = {
                "mean_core": float(np.mean(core_vals)) if len(core_vals) > 0 else np.nan,
                "mean_boundary": float(np.mean(bnd_vals)) if len(bnd_vals) > 0 else np.nan,
            }

            if len(core_vals) >= 2 and len(bnd_vals) >= 2:
                try:
                    stat, pval = mannwhitneyu(core_vals, bnd_vals, alternative="two-sided")
                    entry["mw_stat"] = float(stat)
                    entry["mw_pvalue"] = float(pval)
                except Exception:
                    entry["mw_stat"] = np.nan
                    entry["mw_pvalue"] = np.nan
            else:
                entry["mw_stat"] = np.nan
                entry["mw_pvalue"] = np.nan

            out[regime_key][metric] = entry

    # Overall correlation between boundary_distance and metrics
    corr_results: dict = {}
    for metric in test_metrics:
        valid = merged[[metric, "boundary_distance"]].dropna()
        if len(valid) >= 5:
            try:
                rho, pval = spearmanr(valid["boundary_distance"], valid[metric])
                corr_results[metric] = {"spearman_rho": float(rho), "spearman_pvalue": float(pval)}
            except Exception:
                corr_results[metric] = {"spearman_rho": np.nan, "spearman_pvalue": np.nan}
        else:
            corr_results[metric] = {"spearman_rho": np.nan, "spearman_pvalue": np.nan}

    out["distance_correlations"] = corr_results

    return out
