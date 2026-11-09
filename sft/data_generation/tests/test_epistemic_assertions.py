from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]
from utils.conclusion import check_conclusion, conclusion_payload


class EpistemicAssertionTests(unittest.TestCase):
    def check(self, text):
        atom = {"operator": "AND", "predicate": "have just two species", "claims": [
            {"entity": "Butea", "truth": False, "evidence": [{"source": 1, "quote": text}]},
            {"entity": "Calanthe", "truth": None, "evidence": []}]}
        judge = Mock(side_effect=[atom, {"supported": True},
            {"state": "incomplete", "reason": "Both two and four/five remain possible; Calanthe is unknown."}])
        result = check_conclusion(None, conclusion_payload(
            "Do both Butea and Calanthe have just two species?", [], text), judge)
        return result, judge

    def test_not_definitively_does_not_assert_false_even_if_verifier_says_true(self):
        result, judge = self.check("Butea does not definitively have just two species.")
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(judge.call_count, 3)

    def test_not_necessarily_is_uncertainty(self):
        self.assertEqual(self.check("Butea does not necessarily have just two species.")[0]["state"], "incomplete")

    def test_cannot_confirm_is_not_a_negative_fact(self):
        self.assertEqual(self.check("I cannot confirm Butea has just two species.")[0]["state"], "incomplete")

    def test_definite_numeric_disjunction_still_proves_a_counterexample(self):
        result, judge = self.check("Butea has four or five species, not two.")
        self.assertEqual((result["state"], result["answer"]), ("complete", "No"))
        self.assertEqual(judge.call_count, 2)


if __name__ == "__main__":
    unittest.main()
