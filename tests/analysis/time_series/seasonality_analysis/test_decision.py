"""Tests for decision.py — four-gate framework."""

from automl.analysis.time_series.seasonality_analysis.decision import apply_decision_gates


class TestDecisionGates:
    def test_all_gates_pass(self):
        result = apply_decision_gates(
            step2_pooled={"p_value": 0.001, "pooled_effect_size": 2.0, "r_squared": 0.1},
            step3_stability={"stable": True},
            step3_consistency={"consistent": True},
        )
        assert result["validated"] is True
        assert all(result["gates_passed"].values())

    def test_fails_significance(self):
        result = apply_decision_gates(
            step2_pooled={"p_value": 0.2, "pooled_effect_size": 2.0},
            step3_stability={"stable": True},
            step3_consistency={"consistent": True},
        )
        assert result["validated"] is False
        assert result["gates_passed"]["gate_1_significance"] is False

    def test_fails_magnitude(self):
        result = apply_decision_gates(
            step2_pooled={"p_value": 0.01, "pooled_effect_size": 0.01},
            step3_stability={"stable": True},
            step3_consistency={"consistent": True},
        )
        assert result["validated"] is False
        assert result["gates_passed"]["gate_2_magnitude"] is False

    def test_fails_stability(self):
        result = apply_decision_gates(
            step2_pooled={"p_value": 0.01, "pooled_effect_size": 2.0},
            step3_stability={"stable": False},
            step3_consistency={"consistent": True},
        )
        assert result["validated"] is False
        assert result["gates_passed"]["gate_3_stability"] is False

    def test_fails_consistency(self):
        result = apply_decision_gates(
            step2_pooled={"p_value": 0.01, "pooled_effect_size": 2.0},
            step3_stability={"stable": True},
            step3_consistency={"consistent": False},
        )
        assert result["validated"] is False
        assert result["gates_passed"]["gate_4_consistency"] is False

    def test_rejection_reason_lists_failed_gates(self):
        result = apply_decision_gates(
            step2_pooled={"p_value": 0.5, "pooled_effect_size": 0.01},
            step3_stability={"stable": False},
            step3_consistency={"consistent": False},
        )
        assert "gate_1" in result["rejection_reason"]
        assert "gate_2" in result["rejection_reason"]
