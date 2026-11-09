from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]
from utils.conclusion import check_conclusion, conclusion_payload


class ExplicitIdentityTests(unittest.TestCase):
    def check_fallback(self, question, reasoning):
        judge = Mock(return_value={"state": "incomplete", "reason": "Still missing a relation."})
        result = check_conclusion(None, conclusion_payload(question, [], reasoning), judge)
        self.assertEqual(result["state"], "incomplete")
        judge.assert_called_once()

    def test_definite_requested_actor_is_complete_without_final_message(self):
        q = "Film B stars what American actor whose most famous role is Role A?"
        reasoning = "The actor is Philip Michael Thomas. Now I need to find the connection to Film B; this likely refers to a band."
        judge = Mock()
        result = check_conclusion(None, conclusion_payload(q, [], reasoning), judge)
        self.assertEqual((result["state"], result["answer"]), ("complete", "Philip Michael Thomas"))
        self.assertIn(result["quote"], reasoning)
        judge.assert_not_called()

    def test_actor_identity_is_partial_for_birth_year_question(self):
        self.check_fallback("In what year was the actor in Film B born?",
            "The actor is Philip Michael Thomas. Find the birth year.")

    def test_tentative_identity_does_not_trigger_rule(self):
        self.check_fallback("Which actor starred in Film B?",
            "The actor is probably Philip Michael Thomas. Verify the role.")

    def test_retraction_does_not_trigger_rule(self):
        self.check_fallback("Which actor starred in Film B?",
            "The actor is Philip Michael Thomas. I may be wrong and need to check.")

    def test_negated_identity_does_not_trigger_rule(self):
        self.check_fallback("Which actor starred in Film B?", "The actor is not Philip Michael Thomas.")

    def test_known_anchor_name_cannot_be_treated_as_unknown_answer(self):
        self.check_fallback("Which actor replaced Philip Michael Thomas?",
            "The actor is Philip Michael Thomas. Find his replacement.")

    def test_multiple_requested_unknowns_do_not_trigger_rule(self):
        self.check_fallback("Which actor starred in Film B and when was he born?",
            "The actor is Philip Michael Thomas. Find his birth year.")

    def test_partial_film_relation_does_not_trigger_rule(self):
        self.check_fallback("Which actor starred in Film B and played Role A?",
            "Philip Michael Thomas is in Film B. I still need the Role A link.")


if __name__ == "__main__":
    unittest.main()
