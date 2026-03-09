"""4d + 4e: GARCH volatility modeling and realized variance."""

import numpy as np
import pandas as pd
from arch import arch_model


def compute_garch(target: pd.Series) -> dict:
    """Fit GJR-GARCH(1,1) with fallback to GARCH(1,1).

    Parameters
    ----------
    target : pd.Series
        Target return / change series.

    Returns
    -------
    dict — flat dictionary with GARCH parameters.
    """
    x = target.dropna().values
    out: dict = {}

    if len(x) < 50:
        for key in [
            "garch_omega", "garch_alpha", "garch_beta", "garch_gamma",
            "garch_persistence", "garch_model_type",
        ]:
            out[key] = np.nan
        out["garch_converged"] = False
        out["garch_error"] = True
        return out

    # Scale to percentage returns for numerical stability
    scale = x.std()
    if scale == 0 or np.isnan(scale):
        for key in [
            "garch_omega", "garch_alpha", "garch_beta", "garch_gamma",
            "garch_persistence", "garch_model_type",
        ]:
            out[key] = np.nan
        out["garch_converged"] = False
        out["garch_error"] = True
        return out

    x_scaled = x / scale * 100

    # Try GJR-GARCH(1,1) first
    for model_type, vol_type in [("GJR-GARCH", "GARCH"), ("GARCH", "GARCH")]:
        try:
            o = "har" if model_type == "GJR-GARCH" else "zero"
            am = arch_model(
                x_scaled,
                vol=vol_type,
                p=1,
                q=1,
                o=1 if model_type == "GJR-GARCH" else 0,
                mean="Constant",
                dist="normal",
            )
            res = am.fit(disp="off", show_warning=False)

            if res.convergence_flag == 0:
                params = res.params
                out["garch_omega"] = float(params.get("omega", np.nan))
                out["garch_alpha"] = float(params.get("alpha[1]", np.nan))
                out["garch_beta"] = float(params.get("beta[1]", np.nan))
                out["garch_gamma"] = float(params.get("gamma[1]", 0.0))
                out["garch_persistence"] = float(
                    out["garch_alpha"] + out["garch_beta"] + 0.5 * out["garch_gamma"]
                )
                out["garch_model_type"] = model_type
                out["garch_converged"] = True
                out["garch_error"] = False
                return out
        except Exception:
            continue

    # Both failed
    for key in [
        "garch_omega", "garch_alpha", "garch_beta", "garch_gamma",
        "garch_persistence",
    ]:
        out[key] = np.nan
    out["garch_model_type"] = np.nan
    out["garch_converged"] = False
    out["garch_error"] = True
    return out


def compute_realized_variance(
    target: pd.Series,
    sampling_freq: int,
    horizon: int,
) -> dict:
    """Compute realized variance from high-frequency (non-downsampled) target.

    Uses rolling windows of size *horizon // sampling_freq* on the original
    frequency series, then reports summary statistics of the RV distribution.

    Parameters
    ----------
    target : pd.Series
        Full-frequency target series.
    sampling_freq, horizon : int
        For determining the window size.

    Returns
    -------
    dict — flat dictionary with realized variance stats.
    """
    x = target.dropna()
    k = max(1, horizon // sampling_freq)
    out: dict = {}

    if len(x) < k:
        for key in ["rv_mean", "rv_std", "rv_median", "rv_n"]:
            out[key] = np.nan
        out["rv_error"] = True
        return out

    try:
        squared = x ** 2
        rv = squared.rolling(window=k).sum().dropna()
        out["rv_mean"] = float(rv.mean())
        out["rv_std"] = float(rv.std())
        out["rv_median"] = float(rv.median())
        out["rv_n"] = len(rv)
        out["rv_error"] = False
    except Exception:
        for key in ["rv_mean", "rv_std", "rv_median", "rv_n"]:
            out[key] = np.nan
        out["rv_error"] = True

    return out
