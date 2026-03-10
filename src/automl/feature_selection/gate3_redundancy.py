"""Gate 3: Redundancy detection via Spearman correlation and VIF."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


def apply_gate3(
    panels: dict[str, pd.DataFrame],
    corr_threshold: float = 0.8,
    vif_threshold: float = 10.0,
) -> dict:
    """Detect redundant features via correlation and VIF.

    Parameters
    ----------
    panels : dict[str, pd.DataFrame]
        Winsorised feature panels keyed by feature name.
    corr_threshold : float
        Flag pairs with |ρ| above this.
    vif_threshold : float
        Flag features with VIF above this.

    Returns
    -------
    dict with corr_matrix, high_corr_pairs, vif_scores, flagged_redundant.
    """
    feature_names = sorted(panels.keys())
    n_features = len(feature_names)

    # Stack each feature into a single vector for cross-feature comparison
    stacked: dict[str, np.ndarray] = {}
    for fname in feature_names:
        vals = panels[fname].values.flatten()
        stacked[fname] = vals

    # Pairwise Spearman correlation
    corr_mat = pd.DataFrame(np.nan, index=feature_names, columns=feature_names)
    high_pairs: list[tuple] = []

    for i, fi in enumerate(feature_names):
        corr_mat.loc[fi, fi] = 1.0
        for j in range(i + 1, n_features):
            fj = feature_names[j]
            mask = ~(np.isnan(stacked[fi]) | np.isnan(stacked[fj]))
            if mask.sum() < 20:
                continue
            rho, _ = spearmanr(stacked[fi][mask], stacked[fj][mask])
            corr_mat.loc[fi, fj] = rho
            corr_mat.loc[fj, fi] = rho
            if abs(rho) > corr_threshold:
                high_pairs.append((fi, fj, float(rho)))

    # VIF
    vif_scores: dict[str, float] = {}
    flagged: set = set()

    if n_features >= 2:
        # Build matrix: timestamps × features (use mean across series per timestamp)
        ts_matrix = pd.DataFrame({
            fname: panels[fname].mean(axis=1)
            for fname in feature_names
        }).dropna()

        if len(ts_matrix) > n_features + 1:
            try:
                X = ts_matrix.values
                for i, fname in enumerate(feature_names):
                    try:
                        v = variance_inflation_factor(X, i)
                        vif_scores[fname] = float(v)
                        if v > vif_threshold:
                            flagged.add(fname)
                    except Exception:
                        vif_scores[fname] = np.nan
            except Exception:
                for fname in feature_names:
                    vif_scores[fname] = np.nan
        else:
            for fname in feature_names:
                vif_scores[fname] = np.nan
    else:
        for fname in feature_names:
            vif_scores[fname] = 1.0

    # Also flag features appearing in high-corr pairs
    for fi, fj, _ in high_pairs:
        flagged.add(fi)
        flagged.add(fj)

    return {
        "corr_matrix": corr_mat,
        "high_corr_pairs": high_pairs,
        "vif_scores": vif_scores,
        "flagged_redundant": list(flagged),
    }
