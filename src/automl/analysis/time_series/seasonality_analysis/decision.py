"""Four-gate decision framework for seasonal pattern validation."""

import numpy as np


def apply_decision_gates(
    step2_pooled: dict,
    step3_stability: dict,
    step3_consistency: dict,
    magnitude_threshold: float = 0.5,
    significance_level: float = 0.05,
) -> dict:
    """Evaluate four gates to validate a seasonal candidate.

    Gates
    -----
    1. Statistical significance: pooled p-value < significance_level
    2. Economic magnitude: effect size > magnitude_threshold
    3. Sub-sample stability: stable == True (≥67% splits significant, same direction)
    4. Cross-series consistency: consistent == True (≥60% significant, ≥75% same direction)

    Returns
    -------
    dict with gates_passed, validated, rejection_reason.
    """
    # Gate 1: significance
    p_value = step2_pooled.get("p_value", 1.0)
    gate_1 = bool(p_value < significance_level) if not np.isnan(p_value) else False

    # Gate 2: magnitude
    effect_size = step2_pooled.get("pooled_effect_size", 0.0)
    if effect_size is None or (isinstance(effect_size, float) and np.isnan(effect_size)):
        # Fall back to r_squared or group_means spread
        r_sq = step2_pooled.get("r_squared", 0.0)
        if "group_means" in step2_pooled and step2_pooled["group_means"]:
            vals = list(step2_pooled["group_means"].values())
            effect_size = max(vals) - min(vals)
        elif "coef" in step2_pooled:
            effect_size = abs(step2_pooled["coef"])
        else:
            effect_size = 0.0
    gate_2 = abs(effect_size) > magnitude_threshold

    # Gate 3: stability
    gate_3 = step3_stability.get("stable", False)

    # Gate 4: consistency
    gate_4 = step3_consistency.get("consistent", False)

    gates = {
        "gate_1_significance": gate_1,
        "gate_2_magnitude": gate_2,
        "gate_3_stability": gate_3,
        "gate_4_consistency": gate_4,
    }

    validated = all(gates.values())

    rejection_reason = ""
    if not validated:
        failed = [k for k, v in gates.items() if not v]
        rejection_reason = f"Failed: {', '.join(failed)}"

    return {
        "gates_passed": gates,
        "validated": validated,
        "rejection_reason": rejection_reason,
    }
