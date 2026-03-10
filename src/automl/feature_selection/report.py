"""Report dataclasses for feature selection pipeline."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureResult:
    feature_name: str

    # Gate 1
    gate1_passed: bool = True

    # Gate 2
    gate2_passed: bool = True
    gate2_coverage: float = np.nan
    gate2_adf_reject_frac: float = np.nan
    gate2_hurst_mean: float = np.nan

    # Gate 3
    gate3_flagged_redundant: bool = False
    gate3_max_pairwise_corr: float = np.nan
    gate3_vif: float = np.nan

    # Gate 4
    gate4_passed: bool = False
    gate4_ic_mean: float = np.nan
    gate4_icir: float = np.nan
    gate4_t_stat_nw: float = np.nan
    gate4_p_value_nw: float = np.nan
    gate4_ic_positive_frac: float = np.nan

    # Gate 5
    gate5_passed: bool = False
    gate5_bh_passed: bool = False
    gate5_combined_passed: bool = False

    # Gate 6
    gate6_classification: str = "unconditional"

    # Gate 7
    gate7_half_life: float = np.nan
    gate7_decay_classification: str = "medium"

    # Overall
    all_gates_passed: bool = False
    rejection_reason: str = ""


@dataclass
class FeatureSelectionReport:
    features: dict  # str -> FeatureResult
    selected_features: list
    rejected_features: list
    conditional_features: list
    corr_matrix: Optional[pd.DataFrame] = None
    redundancy_pairs: list = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per feature, sorted by gate4_icir descending."""
        rows = []
        for fname, fr in self.features.items():
            rows.append({
                "feature_name": fr.feature_name,
                "gate1_passed": fr.gate1_passed,
                "gate2_passed": fr.gate2_passed,
                "gate2_coverage": fr.gate2_coverage,
                "gate3_flagged_redundant": fr.gate3_flagged_redundant,
                "gate3_vif": fr.gate3_vif,
                "gate4_passed": fr.gate4_passed,
                "gate4_ic_mean": fr.gate4_ic_mean,
                "gate4_icir": fr.gate4_icir,
                "gate4_t_stat_nw": fr.gate4_t_stat_nw,
                "gate4_ic_positive_frac": fr.gate4_ic_positive_frac,
                "gate5_passed": fr.gate5_passed,
                "gate6_classification": fr.gate6_classification,
                "gate7_half_life": fr.gate7_half_life,
                "gate7_decay_classification": fr.gate7_decay_classification,
                "all_gates_passed": fr.all_gates_passed,
                "rejection_reason": fr.rejection_reason,
            })
        df = pd.DataFrame(rows)
        if "gate4_icir" in df.columns:
            df = df.sort_values("gate4_icir", ascending=False, na_position="last")
        return df.reset_index(drop=True)
