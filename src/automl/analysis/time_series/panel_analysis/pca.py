"""6b: PCA per regime on level panels."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_per_regime(
    panel_level: pd.DataFrame,
    series_regime: pd.Series,
    n_components: int = 5,
    scale: bool = True,
) -> dict:
    """Apply separate PCA to level panels within each series regime.

    Parameters
    ----------
    panel_level : pd.DataFrame
        Rows = timestamps, cols = series_id.
    series_regime : pd.Series
        Index = series_id, values = structural group label.
    n_components : int
        Maximum number of principal components to retain.
    scale : bool
        Whether to standardise columns before PCA.

    Returns
    -------
    dict keyed by regime label, each containing PCA results.
    """
    regimes = series_regime.reindex(panel_level.columns).dropna()
    unique_regimes = sorted(regimes.unique())
    out: dict = {}

    for r in unique_regimes:
        members = regimes[regimes == r].index.tolist()
        sub = panel_level[members].copy()

        # Drop timestamps with > 20% missing
        thresh = int(len(members) * 0.8)
        sub = sub.dropna(thresh=thresh)
        sub = sub.dropna(axis=1, how="all")

        if sub.shape[0] < 10 or sub.shape[1] < 2:
            out[str(r)] = {
                "explained_variance_ratio": np.array([]),
                "cumulative_variance": np.array([]),
                "pc1_variance": np.nan,
                "loadings": pd.DataFrame(),
                "factor_scores": pd.DataFrame(),
                "n_series": sub.shape[1],
                "error": True,
            }
            continue

        # Fill remaining NaNs with column means for PCA
        sub = sub.fillna(sub.mean())
        cols = sub.columns.tolist()

        X = sub.values
        if scale:
            X = StandardScaler().fit_transform(X)

        nc = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=nc)
        scores = pca.fit_transform(X)

        evr = pca.explained_variance_ratio_
        out[str(r)] = {
            "explained_variance_ratio": evr,
            "cumulative_variance": np.cumsum(evr),
            "pc1_variance": float(evr[0]),
            "loadings": pd.DataFrame(
                pca.components_.T,
                index=cols,
                columns=[f"PC{i+1}" for i in range(nc)],
            ),
            "factor_scores": pd.DataFrame(
                scores,
                index=sub.index,
                columns=[f"PC{i+1}" for i in range(nc)],
            ),
            "n_series": len(cols),
            "error": False,
        }

    return out
