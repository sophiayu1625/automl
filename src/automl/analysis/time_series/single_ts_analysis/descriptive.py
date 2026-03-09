"""4a: Descriptive statistics — overall and per-regime."""

import numpy as np
import pandas as pd

_PERCENTILES = [1, 5, 25, 75, 95, 99]


def _stats_for_array(x: np.ndarray, prefix: str) -> dict:
    """Compute descriptive stats for a 1-d array and return as flat dict."""
    if len(x) < 2:
        keys = ["n", "mean", "std", "skew", "kurt"] + [f"p{p}" for p in _PERCENTILES]
        return {f"{prefix}_{k}": np.nan for k in keys}

    pcts = np.percentile(x, _PERCENTILES)
    return {
        f"{prefix}_n": len(x),
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x, ddof=1),
        f"{prefix}_skew": float(pd.Series(x).skew()),
        f"{prefix}_kurt": float(pd.Series(x).kurt()),
        **{f"{prefix}_p{p}": float(v) for p, v in zip(_PERCENTILES, pcts)},
    }


def compute_descriptive_stats(
    target: pd.Series,
    regime_labels: pd.Series | None = None,
    regime_names: dict[int, str] | None = None,
) -> dict:
    """Return overall + per-regime descriptive statistics.

    Parameters
    ----------
    target : pd.Series
        The target return / change series.
    regime_labels : pd.Series, optional
        Integer regime labels aligned to *target*.
    regime_names : dict, optional
        Mapping from regime int -> human-readable name (used in keys).
    """
    out = _stats_for_array(target.dropna().values, prefix="desc")

    if regime_labels is not None:
        aligned = pd.DataFrame({"target": target, "regime": regime_labels}).dropna()
        for rid in sorted(aligned["regime"].unique()):
            name = regime_names.get(int(rid), str(int(rid))) if regime_names else str(int(rid))
            subset = aligned.loc[aligned["regime"] == rid, "target"].values
            out.update(_stats_for_array(subset, prefix=f"desc_r{name}"))

    return out
