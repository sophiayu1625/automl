"""Main entry point: run_panel_analysis() orchestrates all panel sub-analyses."""

import pandas as pd

from automl.analysis.time_series.panel_analysis.correlation import compute_correlation_matrix
from automl.analysis.time_series.panel_analysis.pca import compute_pca_per_regime
from automl.analysis.time_series.panel_analysis.rolling_correlation import compute_rolling_correlation
from automl.analysis.time_series.panel_analysis.cross_regime import compare_regimes
from automl.analysis.time_series.panel_analysis.boundary import analyse_boundary_effects


def run_panel_analysis(
    panel_target: pd.DataFrame,
    panel_level: pd.DataFrame,
    series_regime: pd.Series,
    macro_regime: pd.Series,
    results_df: pd.DataFrame,
    series_metadata: pd.DataFrame,
    sampling_freq: str = "1h",
    horizon: str = "24h",
    rolling_window: str = "63D",
    pca_n_components: int = 5,
) -> dict:
    """Run full panel analysis and return results dictionary.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows = timestamps, cols = series_id; target values.
    panel_level : pd.DataFrame
        Rows = timestamps, cols = series_id; level values.
    series_regime : pd.Series
        Index = series_id, values = structural group label (fixed per series).
    macro_regime : pd.Series
        Index = timestamp, values = time-varying regime label.
    results_df : pd.DataFrame
        Phase 4 single-series output; index = series_id.
    series_metadata : pd.DataFrame
        Index = series_id; columns include 'maturity', 'boundary_distance'.
    sampling_freq : str
        Sampling frequency string.
    horizon : str
        Prediction horizon string.
    rolling_window : str
        Window for rolling correlation (default '63D').
    pca_n_components : int
        Max PCA components to retain.

    Returns
    -------
    dict with keys: correlation, pca, rolling_correlation,
         cross_regime, architecture_recommendation, boundary.
    """
    # 6a — Correlation matrix
    corr_results = compute_correlation_matrix(panel_target, series_regime)

    # 6b — PCA per regime
    pca_results = compute_pca_per_regime(
        panel_level, series_regime, n_components=pca_n_components,
    )

    # 6c — Rolling correlation
    rolling_corr_df = compute_rolling_correlation(
        panel_target, series_regime, rolling_window=rolling_window,
    )

    # 6d — Cross-regime comparison
    cross_regime_results = compare_regimes(panel_target, series_regime, results_df)

    # 6e — Boundary effects
    boundary_results = analyse_boundary_effects(
        results_df, series_metadata, series_regime,
    )

    return {
        "correlation": corr_results,
        "pca": pca_results,
        "rolling_correlation": rolling_corr_df,
        "cross_regime": cross_regime_results,
        "architecture_recommendation": cross_regime_results.get(
            "architecture_recommendation", "investigate_further"
        ),
        "boundary": boundary_results,
    }
