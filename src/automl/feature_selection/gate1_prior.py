"""Gate 1: Economic prior filter."""


def apply_gate1(
    feature_names: list[str],
    approved: list[str] | None = None,
) -> dict[str, bool]:
    """Filter features by economic prior approval.

    Parameters
    ----------
    feature_names : list[str]
        All candidate feature names.
    approved : list[str] | None
        Pre-approved feature names. If None, all pass (deferred review).

    Returns
    -------
    dict — {feature_name: passed_bool}.
    """
    if approved is None:
        return {f: True for f in feature_names}
    approved_set = set(approved)
    return {f: (f in approved_set) for f in feature_names}
