"""Seasonal component estimation and adjustment."""

import numpy as np
import pandas as pd


def estimate_seasonal_component(
    panel_target: pd.DataFrame,
    grouping_var: pd.Series,
    candidate_type: str = "categorical",
    method: str = "group_mean",
) -> pd.DataFrame:
    """Estimate and subtract seasonal component from the panel.

    Parameters
    ----------
    panel_target : pd.DataFrame
        Rows = timestamps, cols = series_id.
    grouping_var : pd.Series
        Grouping labels aligned to panel index.
    candidate_type : str
        'categorical' or 'continuous'.
    method : str
        'group_mean', 'regression', or 'stl'.

    Returns
    -------
    pd.DataFrame — seasonal component (same shape as panel_target).
    """
    seasonal = pd.DataFrame(0.0, index=panel_target.index, columns=panel_target.columns)
    aligned_group = grouping_var.reindex(panel_target.index)

    if method == "group_mean" and candidate_type == "categorical":
        for col in panel_target.columns:
            series = panel_target[col]
            df = pd.DataFrame({"value": series, "group": aligned_group}).dropna()
            if len(df) < 10:
                continue
            group_means = df.groupby("group")["value"].mean()
            seasonal[col] = aligned_group.map(group_means).fillna(0.0)

    elif method == "regression" or candidate_type == "continuous":
        for col in panel_target.columns:
            series = panel_target[col]
            df = pd.DataFrame({"y": series, "x": aligned_group}).dropna()
            if len(df) < 10:
                continue
            x = df["x"].values
            y = df["y"].values
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2)
            if denom > 0:
                beta = np.sum((x - x_mean) * (y - y_mean)) / denom
                alpha = y_mean - beta * x_mean
                seasonal[col] = alpha + beta * aligned_group.fillna(0.0)

    elif method == "stl":
        try:
            from statsmodels.tsa.seasonal import STL

            for col in panel_target.columns:
                series = panel_target[col].dropna()
                if len(series) < 50:
                    continue
                try:
                    stl = STL(series, period=int(grouping_var.nunique()))
                    result = stl.fit()
                    seasonal[col] = result.seasonal.reindex(panel_target.index).fillna(0.0)
                except Exception:
                    # Fallback to group_mean
                    df = pd.DataFrame({"value": series, "group": aligned_group}).dropna()
                    if len(df) >= 10:
                        group_means = df.groupby("group")["value"].mean()
                        seasonal[col] = aligned_group.map(group_means).fillna(0.0)
        except ImportError:
            # STL not available, fallback
            return estimate_seasonal_component(
                panel_target, grouping_var, candidate_type, method="group_mean",
            )

    return seasonal
