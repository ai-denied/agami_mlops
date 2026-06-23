import copy
import unittest

from context_emotion.evaluation import promotion_gate
from context_emotion.tests.fixtures import mock_evaluation_result


def _policy(**overrides):
    policy = {
        "overall_quality_gate": {"required": True, "min_macro_f1": 0.5, "min_accuracy": None, "max_macro_f1_drop_vs_current": None},
        "class_regression_gate": {"required": True, "critical_classes": ["embarrassment", "aversion"],
                                   "max_f1_drop_absolute": 0.05, "max_f1_drop_relative_pct": None},
        "human_ambiguity_gate": {"required": False, "min_exposures_required": None,
                                  "max_ambiguous_rate": None, "max_ambiguous_rate_increase": None},
        "attacker_proxy_gate": {"required": False, "max_attacker_solve_rate": None, "max_attacker_solve_rate_increase": None},
        "artifact_integrity_gate": {"required": True},
    }
    for gate, kv in overrides.items():
        policy[gate].update(kv)
    return policy


class TestPromotionGate(unittest.TestCase):
    def test_clean_candidate_is_promoted(self):
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.70)

        decision = promotion_gate.decide("v0", "v1", current, candidate, _policy())

        self.assertEqual(decision["final_decision"], "promote")
        self.assertEqual(decision["gates"]["overall_quality"], "pass")
        self.assertEqual(decision["gates"]["class_regression"], "pass")
        self.assertEqual(decision["gates"]["artifact_integrity"], "pass")

    def test_class_regression_on_critical_class_rejects(self):
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result(
            "v1", overall_macro_f1=0.70,
            per_class_overrides={"embarrassment": {"precision": 0.3, "recall": 0.3, "f1": 0.3, "support": 10}},
        )

        decision = promotion_gate.decide("v0", "v1", current, candidate, _policy())

        self.assertEqual(decision["gates"]["class_regression"], "fail")
        self.assertEqual(decision["final_decision"], "reject")

    def test_low_overall_quality_rejects(self):
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.10)

        decision = promotion_gate.decide("v0", "v1", current, candidate, _policy())

        self.assertEqual(decision["gates"]["overall_quality"], "fail")
        self.assertEqual(decision["final_decision"], "reject")

    def test_unconfigured_required_gate_forces_manual_review(self):
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.70)
        policy = _policy(class_regression_gate={"max_f1_drop_absolute": None, "max_f1_drop_relative_pct": None})

        decision = promotion_gate.decide("v0", "v1", current, candidate, policy)

        self.assertEqual(decision["gates"]["class_regression"], "not_configured")
        self.assertEqual(decision["final_decision"], "manual_review")

    def test_non_required_gate_failure_does_not_block_promotion(self):
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.70)
        candidate = copy.deepcopy(candidate)
        candidate["attacker_proxy"] = {"status": "available", "attacker_solve_rate": 0.99, "error_type_breakdown": {}, "proxy_model_version": "mock"}
        policy = _policy(attacker_proxy_gate={"required": False, "max_attacker_solve_rate": 0.5})

        decision = promotion_gate.decide("v0", "v1", current, candidate, policy)

        self.assertEqual(decision["gates"]["attacker_proxy_resistance"], "fail")
        self.assertEqual(decision["final_decision"], "promote")  # not required -> informational only

    def test_artifact_integrity_failure_rejects_regardless_of_metrics(self):
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.90)
        candidate["artifact_integrity"]["label_schema_match"] = False

        decision = promotion_gate.decide("v0", "v1", current, candidate, _policy())

        self.assertEqual(decision["gates"]["artifact_integrity"], "fail")
        self.assertEqual(decision["final_decision"], "reject")

    def test_attacker_proxy_not_configured_promotes_but_warns_loudly(self):
        """The default policy ships attacker_proxy_gate.required=false (no
        proxy model/eval pool exists yet - see promotion_policy.yaml TODOs).
        That must not silently look like a clean pass: promotion proceeds,
        but a human reading the decision must see, unambiguously, that
        adversarial robustness was never checked for this promotion."""
        current = mock_evaluation_result("v0", overall_macro_f1=0.65)
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.70)

        decision = promotion_gate.decide("v0", "v1", current, candidate, _policy())

        self.assertEqual(decision["gates"]["attacker_proxy_resistance"], "not_configured")
        self.assertEqual(decision["final_decision"], "promote")
        self.assertTrue(
            any("attacker_proxy_resistance" in w for w in decision["warnings"]),
            f"expected a warning about the unchecked attacker_proxy gate, got: {decision['warnings']}",
        )

    def test_no_current_model_yet_still_evaluates_candidate_alone(self):
        candidate = mock_evaluation_result("v1", overall_macro_f1=0.70)
        decision = promotion_gate.decide("none", "v1", None, candidate, _policy())
        self.assertIn(decision["final_decision"], ("promote", "manual_review", "reject"))
        self.assertEqual(decision["gates"]["overall_quality"], "pass")


if __name__ == "__main__":
    unittest.main()
