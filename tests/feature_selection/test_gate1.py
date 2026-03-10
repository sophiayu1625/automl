"""Tests for gate1_prior.py."""

from automl.feature_selection.gate1_prior import apply_gate1


class TestGate1:
    def test_none_approved_all_pass(self):
        result = apply_gate1(["a", "b", "c"], approved=None)
        assert all(result.values())

    def test_subset_approved(self):
        result = apply_gate1(["a", "b", "c"], approved=["a", "c"])
        assert result["a"] is True
        assert result["b"] is False
        assert result["c"] is True

    def test_empty_approved(self):
        result = apply_gate1(["a", "b"], approved=[])
        assert result["a"] is False
