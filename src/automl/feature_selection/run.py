"""Main entry point: run_feature_selection()."""

import numpy as np
import pandas as pd

from automl.feature_selection.gate1_prior import apply_gate1
from automl.feature_selection.gate2_quality import apply_gate2
from automl.feature_selection.gate3_redundancy import apply_gate3
from automl.feature_selection.gate4_ic import (
    compute_cross_sectional_ic,
    compute_ic_stats,
    apply_gate4_decision,
)
from automl.feature_selection.gate5_multiple import apply_gate5
from automl.feature_selection.gate6_regime import compute_regime_ic
from automl.feature_selection.gate7_decay import compute_ic_decay
from automl.feature_selection.report import (
    FeatureResult,
    FeatureSelectionReport,
)


def run_feature_selection(
    panel_target: pd.DataFrame,
    panel_features: dict[str, pd.DataFrame],
    series_regime: pd.Series,
    macro_regime: pd.Series,
    sampling_freq: str = "1h",
    horizon: str = "24h",
    ic_window: str = "63D",
    ic_min_periods: int = 30,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.8,
    ic_mean_threshold: float = 0.02,
    icir_threshold: float = 0.5,
    ic_tstat_threshold: float = 2.0,
    ic_positive_frac_threshold: float = 0.55,
    fdr_q: float = 0.10,
    ic_decay_lags: list[int] | None = None,
    significance_level: float = 0.05,
    gate1_approved: list[str] | None = None,
) -> FeatureSelectionReport:
    """Run seven-gate feature selection pipeline.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows=timestamps, cols=series_id.
    panel_features : dict[str, pd.DataFrame]
        Keyed by feature name; each panel same shape as panel_target.
    series_regime : pd.Series
        Index=series_id, values=structural group label.
    macro_regime : pd.Series
        Index=timestamp, values=time-varying regime label.
    gate1_approved : list[str] | None
        Pre-approved features. None = all pass.

    Returns
    -------
    FeatureSelectionReport
    """
    if ic_decay_lags is None:
        ic_decay_lags = [1, 2, 5, 10, 21]

    feature_names = sorted(panel_features.keys())
    nw_lags = max(1, _parse_horizon_ratio(horizon, sampling_freq))

    # Gate 1
    gate1 = apply_gate1(feature_names, gate1_approved)

    # Gate 2
    gate2_out = apply_gate2(panel_features, significance_level)
    gate2_results = gate2_out["results"]
    panels = gate2_out["winsorised_panels"]

    # Gate 3
    gate3 = apply_gate3(panels, corr_threshold, vif_threshold)

    # Gate 4: IC computation
    ic_stats_all: dict[str, dict] = {}
    for fname in feature_names:
        ic_series = compute_cross_sectional_ic(panel_target, panels[fname])
        ic_stats_all[fname] = compute_ic_stats(ic_series, nw_lags)

    gate4_decisions = {
        fname: apply_gate4_decision(
            ic_stats_all[fname], ic_mean_threshold, icir_threshold,
            ic_tstat_threshold, ic_positive_frac_threshold,
        )
        for fname in feature_names
    }

    # Gate 5
    gate5 = apply_gate5(
        ic_stats_all, fdr_q, ic_mean_threshold, icir_threshold, ic_tstat_threshold,
    )

    # Gate 6 & 7 (only for features that passed gates 1-5)
    gate6_results: dict[str, dict] = {}
    gate7_results: dict[str, dict] = {}

    for fname in feature_names:
        passes_so_far = (
            gate1[fname]
            and gate2_results[fname]["passed"]
            and gate4_decisions[fname]["passed"]
            and gate5[fname]["gate5_passed"]
        )
        if passes_so_far:
            gate6_results[fname] = compute_regime_ic(
                panel_target, panels[fname], series_regime, macro_regime, nw_lags,
            )
            gate7_results[fname] = compute_ic_decay(
                panel_target, panels[fname], ic_decay_lags, nw_lags,
            )
        else:
            gate6_results[fname] = {"classification": "unconditional"}
            gate7_results[fname] = {
                "half_life": np.nan, "decay_classification": "medium",
            }

    # Build FeatureResults
    features: dict[str, FeatureResult] = {}
    for fname in feature_names:
        g2 = gate2_results[fname]
        ics = ic_stats_all[fname]
        g4 = gate4_decisions[fname]
        g5 = gate5[fname]
        g6 = gate6_results[fname]
        g7 = gate7_results[fname]

        # Max pairwise corr for this feature
        corr_row = gate3["corr_matrix"].loc[fname].drop(fname, errors="ignore")
        max_corr = float(corr_row.abs().max()) if len(corr_row) > 0 else np.nan

        # Determine first failing gate
        rejection = ""
        if not gate1[fname]:
            rejection = "gate1_economic_prior"
        elif not g2["passed"]:
            rejection = "gate2_data_quality"
        elif not g4["passed"]:
            rejection = "gate4_ic_insufficient"
        elif not g5["gate5_passed"]:
            rejection = "gate5_multiple_testing"
        elif g6.get("classification") == "dangerous":
            rejection = "gate6_regime_sign_flip"

        all_passed = (
            gate1[fname]
            and g2["passed"]
            and g4["passed"]
            and g5["gate5_passed"]
            and g6.get("classification") != "dangerous"
        )

        features[fname] = FeatureResult(
            feature_name=fname,
            gate1_passed=gate1[fname],
            gate2_passed=g2["passed"],
            gate2_coverage=g2["coverage"],
            gate2_adf_reject_frac=g2["adf_reject_frac"],
            gate2_hurst_mean=g2["hurst_mean"],
            gate3_flagged_redundant=fname in gate3["flagged_redundant"],
            gate3_max_pairwise_corr=max_corr,
            gate3_vif=gate3["vif_scores"].get(fname, np.nan),
            gate4_passed=g4["passed"],
            gate4_ic_mean=ics["ic_mean"],
            gate4_icir=ics["icir"],
            gate4_t_stat_nw=ics["t_stat_nw"],
            gate4_p_value_nw=ics["p_value_nw"],
            gate4_ic_positive_frac=ics["ic_positive_frac"],
            gate5_passed=g5["gate5_passed"],
            gate5_bh_passed=g5["bh_passed"],
            gate5_combined_passed=g5["combined_passed"],
            gate6_classification=g6.get("classification", "unconditional"),
            gate7_half_life=g7.get("half_life", np.nan),
            gate7_decay_classification=g7.get("decay_classification", "medium"),
            all_gates_passed=all_passed,
            rejection_reason=rejection,
        )

    selected = [f for f, fr in features.items() if fr.all_gates_passed]
    rejected = [f for f, fr in features.items() if not fr.all_gates_passed]
    conditional = [
        f for f, fr in features.items()
        if fr.all_gates_passed and fr.gate6_classification != "unconditional"
    ]

    return FeatureSelectionReport(
        features=features,
        selected_features=selected,
        rejected_features=rejected,
        conditional_features=conditional,
        corr_matrix=gate3["corr_matrix"],
        redundancy_pairs=gate3["high_corr_pairs"],
    )


def _parse_horizon_ratio(horizon: str, sampling_freq: str) -> int:
    """Parse horizon/sampling_freq ratio from strings like '24h'/'1h'."""
    try:
        h = pd.Timedelta(horizon)
        s = pd.Timedelta(sampling_freq)
        return max(1, int(h / s))
    except Exception:
        return 1
