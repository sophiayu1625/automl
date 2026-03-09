"""Parallel runner: run_all(list_of_inputs) -> pd.DataFrame."""

from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from automl.analysis.time_series.single_ts_analysis.analyse import analyse_series


def run_all(
    inputs: list[dict[str, Any]],
    n_jobs: int = -1,
    verbose: int = 0,
) -> pd.DataFrame:
    """Run analyse_series() in parallel across multiple series.

    Parameters
    ----------
    inputs : list[dict]
        Each dict contains the keyword arguments for ``analyse_series()``.
        Required keys: ``series_id``, ``target``, ``level``.
        Optional keys: ``regime_labels``, ``sampling_freq``, ``horizon``,
        ``regime_names``.
    n_jobs : int
        Number of parallel workers (default -1 = all cores).
    verbose : int
        Joblib verbosity level.

    Returns
    -------
    pd.DataFrame — one row per series, columns from all analysis results.
    """
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(analyse_series)(**inp) for inp in inputs
    )
    return pd.DataFrame(results)
