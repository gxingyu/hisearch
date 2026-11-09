from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]
from utils.conclusion import check_conclusion, conclusion_payload


class AssertedBooleanTests(unittest.TestCase):
    def payload(self, text):
        return conclusion_payload("Are P and Q both economic-based documentaries?", [], text)

    def atoms(self, truth, quote, operator="AND", source=1):
        return {"operator": operator, "predicate": "economic-based documentaries", "claims": [
            {"entity": "P", "truth": truth, "evidence": [{"source": source, "quote": quote}]},
            {"entity": "Q", "truth": None, "evidence": []}]}

    def test_unproven_negative_assertion_already_commits_to_no(self):
        text = "P is about violence, not economics. I still need Q."
        judge = Mock(side_effect=[self.atoms(False, "P is about violence, not economics."), {"supported": True}])
        result = check_conclusion(None, self.payload(text), judge)
        self.assertEqual((result["state"], result["answer"]), ("complete", "No"))
        self.assertIn(result["quote"], text)
        extraction = judge.call_args_list[0].args[1]
        self.assertNotIn("retrieved_fact_secret", extraction)
        self.assertNotIn('"question"', judge.call_args_list[1].args[1].split("INPUT:\n")[1])

    def test_one_positive_and_unknown_keeps_necessary_other_subject(self):
        text = "P is economic-based. I need Q."
        judge = Mock(side_effect=[self.atoms(True, "P is economic-based."),
            {"state": "incomplete", "reason": "Q is still unknown."}])
        self.assertEqual(check_conclusion(None, self.payload(text), judge)["state"], "incomplete")

    def test_one_asserted_or_positive_commits_to_yes(self):
        text = "P is economic-based. I need Q."
        payload = conclusion_payload("Is either P or Q economic-based documentaries?", [], text)
        judge = Mock(side_effect=[self.atoms(True, "P is economic-based.", "OR"), {"supported": True}])
        self.assertEqual(check_conclusion(None, payload, judge)["answer"], "Yes")

    def test_invalid_claim_quote_cannot_create_a_conclusion(self):
        judge = Mock(side_effect=[self.atoms(False, "Invented negative."),
            {"state": "incomplete", "reason": "No definite claim."}])
        self.assertEqual(check_conclusion(None, self.payload("Check P and Q."), judge)["state"], "incomplete")

    def test_tentative_claim_rejected_by_assertion_verifier(self):
        text = "P might not be economic-based. Check it."
        judge = Mock(side_effect=[self.atoms(False, text), {"supported": False},
            {"state": "incomplete", "reason": "Tentative only."}])
        self.assertEqual(check_conclusion(None, self.payload(text), judge)["state"], "incomplete")

    def test_retraction_leaves_full_history_semantic_check_in_control(self):
        judge = Mock(return_value={"state": "incomplete", "reason": "Prior claim retracted."})
        result = check_conclusion(None, self.payload("I was wrong that P is not economic-based. Check P."), judge)
        self.assertEqual(result["state"], "incomplete")
        judge.assert_called_once()

    def test_question_cannot_supply_an_assistant_assertion(self):
        payload = self.payload("Check the documentaries.")
        atom = self.atoms(False, payload["question"], source=0)
        judge = Mock(side_effect=[atom, {"supported": True},
            {"state": "incomplete", "reason": "No reasoning assertion."}])
        self.assertEqual(check_conclusion(None, payload, judge)["state"], "incomplete")


if __name__ == "__main__":
    unittest.main()
