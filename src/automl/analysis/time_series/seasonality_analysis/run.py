"""Main entry point: run_seasonality_analysis()."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from automl.analysis.time_series.seasonality_analysis.candidates import SeasonalCandidate
from automl.analysis.time_series.seasonality_analysis.step1_visual import compute_grouped_stats
from automl.analysis.time_series.seasonality_analysis.step2_tests import (
    check_categorical_seasonality,
    check_continuous_seasonality,
    check_acf_at_seasonal_lag,
    compute_periodogram,
)
from automl.analysis.time_series.seasonality_analysis.step3_stability import (
    check_stability,
    compute_cross_series_consistency,
)
from automl.analysis.time_series.seasonality_analysis.decision import apply_decision_gates
from automl.analysis.time_series.seasonality_analysis.adjustment import estimate_seasonal_component


@dataclass
class CandidateResult:
    name: str
    validated: bool
    rejection_reason: str
    gates: dict
    grouped_stats: pd.DataFrame
    pooled_f_stat: Optional[float]
    pooled_p_value: float
    pooled_r_squared: float
    pooled_effect_size: float
    per_series_results: dict
    acf_at_period: Optional[float]
    acf_significant: Optional[bool]
    stability: dict
    cross_series_pct_significant: float
    cross_series_pct_same_direction: float
    adjustment_method: Optional[str] = None
    seasonal_component: Optional[pd.DataFrame] = None
    adjusted_panel_target: Optional[pd.DataFrame] = None


@dataclass
class SeasonalityReport:
    candidates: dict  # name -> CandidateResult
    validated_candidates: list
    rejected_candidates: list
    periodogram: dict
    recommendation: str


def run_seasonality_analysis(
    panel_target: pd.DataFrame,
    series_regime: pd.Series,
    sampling_freq: str = "1h",
    horizon: str = "24h",
    candidates: list[SeasonalCandidate] | None = None,
    n_stability_splits: int = 3,
    cross_series_threshold: float = 0.6,
    magnitude_threshold_bps: float = 0.5,
    significance_level: float = 0.05,
) -> SeasonalityReport:
    """Run full seasonality validation across all candidates.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows = timestamps, cols = series_id.
    series_regime : pd.Series
        Index = series_id, values = structural group label.
    sampling_freq : str
        Sampling frequency string.
    horizon : str
        Prediction horizon string.
    candidates : list[SeasonalCandidate]
        Seasonal patterns to test.
    n_stability_splits : int
        Number of sub-periods for stability test.
    cross_series_threshold : float
        Min fraction of series showing significance.
    magnitude_threshold_bps : float
        Min effect size for Gate 2.
    significance_level : float
        Alpha for all tests.

    Returns
    -------
    SeasonalityReport
    """
    if candidates is None:
        candidates = []

    # Periodogram (once, across all series)
    periodogram = compute_periodogram(panel_target, sampling_freq)

    results: dict[str, CandidateResult] = {}
    validated = []
    rejected = []

    for cand in candidates:
        # Step 1: grouped stats (pooled across all series)
        pooled_series = panel_target.stack()
        pooled_series.name = "value"
        grouping_aligned = cand.grouping_var.reindex(
            pooled_series.index.get_level_values(0)
        )
        # For stacked data, use timestamp-level grouping
        flat_series = pd.Series(pooled_series.values, index=pooled_series.index.get_level_values(0))
        grouped_stats = compute_grouped_stats(flat_series, grouping_aligned)

        # Step 2: pooled test
        if cand.candidate_type == "categorical":
            pooled_test = check_categorical_seasonality(
                flat_series, grouping_aligned, significance_level,
            )
            pooled_f_stat = pooled_test.get("f_stat")
            effect_size = 0.0
            gm = pooled_test.get("group_means", {})
            if gm:
                vals = list(gm.values())
                effect_size = max(vals) - min(vals) if vals else 0.0
            pooled_test["pooled_effect_size"] = effect_size
        else:
            pooled_test = check_continuous_seasonality(
                flat_series, grouping_aligned, significance_level,
            )
            pooled_f_stat = None
            effect_size = abs(pooled_test.get("coef", 0.0))
            pooled_test["pooled_effect_size"] = effect_size

        # ACF at seasonal lag
        acf_result = {"acf_at_period": None, "acf_significant": None}
        if cand.seasonal_period is not None:
            # Test on first series as representative
            first_col = panel_target.columns[0]
            acf_result = check_acf_at_seasonal_lag(
                panel_target[first_col], cand.seasonal_period, significance_level,
            )

        # Step 2: per-series tests
        per_series_results = {}
        for col in panel_target.columns:
            col_series = panel_target[col].dropna()
            col_group = cand.grouping_var.reindex(col_series.index)
            if cand.candidate_type == "categorical":
                per_series_results[col] = check_categorical_seasonality(
                    col_series, col_group, significance_level,
                )
            else:
                per_series_results[col] = check_continuous_seasonality(
                    col_series, col_group, significance_level,
                )

        # Step 3: stability (on pooled)
        stability = check_stability(
            flat_series, grouping_aligned, cand,
            n_splits=n_stability_splits,
            significance_level=significance_level,
        )

        # Step 3: cross-series consistency
        consistency = compute_cross_series_consistency(
            per_series_results, significance_level,
        )

        # Decision gates
        gates_result = apply_decision_gates(
            step2_pooled=pooled_test,
            step3_stability=stability,
            step3_consistency=consistency,
            magnitude_threshold=magnitude_threshold_bps,
            significance_level=significance_level,
        )

        # If validated, estimate seasonal component
        seasonal_component = None
        adjusted_panel = None
        adjustment_method = None
        if gates_result["validated"]:
            adjustment_method = "group_mean" if cand.candidate_type == "categorical" else "regression"
            seasonal_component = estimate_seasonal_component(
                panel_target, cand.grouping_var, cand.candidate_type, adjustment_method,
            )
            adjusted_panel = panel_target - seasonal_component

        cr = CandidateResult(
            name=cand.name,
            validated=gates_result["validated"],
            rejection_reason=gates_result["rejection_reason"],
            gates=gates_result["gates_passed"],
            grouped_stats=grouped_stats,
            pooled_f_stat=pooled_f_stat,
            pooled_p_value=pooled_test.get("p_value", np.nan),
            pooled_r_squared=pooled_test.get("r_squared", np.nan),
            pooled_effect_size=effect_size,
            per_series_results=per_series_results,
            acf_at_period=acf_result.get("acf_at_period"),
            acf_significant=acf_result.get("acf_significant"),
            stability=stability,
            cross_series_pct_significant=consistency.get("pct_significant", np.nan),
            cross_series_pct_same_direction=consistency.get("pct_same_direction", np.nan),
            adjustment_method=adjustment_method,
            seasonal_component=seasonal_component,
            adjusted_panel_target=adjusted_panel,
        )

        results[cand.name] = cr
        if cr.validated:
            validated.append(cand.name)
        else:
            rejected.append(cand.name)

    # Build recommendation
    if validated:
        rec = f"Remove seasonal components: {', '.join(validated)}. Retain as features or ignore: {', '.join(rejected)}."
    else:
        rec = "No seasonal patterns validated. No adjustment needed."

    return SeasonalityReport(
        candidates=results,
        validated_candidates=validated,
        rejected_candidates=rejected,
        periodogram=periodogram,
        recommendation=rec,
    )
