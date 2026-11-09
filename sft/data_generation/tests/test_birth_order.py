import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]
from utils.birth_order import birth_order_hint, check_birth_order
from utils.readiness import check_ready


class BirthBoundsTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"question": "Between Jon Paul Puno and Terry O. Morse, who was born earlier?",
            "sources": [{"source": 1, "text": "Jon Paul Puno was born in 1985."},
                        {"source": 2, "text": "A 1940 film was directed by Terry O. Morse."}],
            "proposed_reasoning": "Find Terry O. Morse's exact birthday."}
        self.extracted = {"people": [
            {"entity": "Jon Paul Puno", "bounds": [{"relation": "born_in", "year": 1985,
                "evidence": [{"source": 1, "quote": self.payload["sources"][0]["text"]}]}]},
            {"entity": "Terry O. Morse", "bounds": [{"relation": "born_by", "year": 1940,
                "evidence": [{"source": 2, "quote": self.payload["sources"][1]["text"]}]}]}]}

    def test_verified_activity_upper_bound_determines_order(self):
        judge = Mock(side_effect=[self.extracted, {"supported": True}, {"supported": True}])
        result = check_birth_order(None, self.payload, judge)
        self.assertEqual((result["status"], result["answer"]), ("stop", "Terry O. Morse"))
        atom = json.loads(judge.call_args_list[1].args[1].split("INPUT:\n")[1])
        self.assertNotIn("question", atom)
        self.assertNotIn("Jon Paul Puno", json.dumps(atom))

    def test_overlapping_bound_preserves_needed_search(self):
        self.payload["sources"][1]["text"] = "A 2000 film was directed by Terry O. Morse."
        bound = self.extracted["people"][1]["bounds"][0]
        bound.update(year=2000, evidence=[{"source": 2, "quote": self.payload["sources"][1]["text"]}])
        judge = Mock(return_value=self.extracted)
        self.assertEqual(check_ready(None, self.payload, judge)["status"], "missing")
        judge.assert_called_once()

    def test_unknown_other_subject_cannot_be_ordered(self):
        self.extracted["people"][1]["bounds"] = []
        result = check_birth_order(None, self.payload, Mock(return_value=self.extracted))
        self.assertEqual(result["status"], "continue")
        self.assertNotIn("answer", result)

    def test_same_year_bound_cannot_prove_strict_order(self):
        self.payload["sources"][1]["text"] = "A 1985 film was directed by Terry O. Morse."
        bound = self.extracted["people"][1]["bounds"][0]
        bound.update(year=1985, evidence=[{"source": 2, "quote": self.payload["sources"][1]["text"]}])
        self.assertEqual(check_birth_order(None, self.payload, Mock(return_value=self.extracted))["status"], "continue")

    def test_year_must_be_present_in_exact_visible_quote(self):
        self.extracted["people"][1]["bounds"][0]["year"] = 1906
        result = check_birth_order(None, self.payload, Mock(return_value=self.extracted))
        self.assertEqual(result["status"], "continue")

    def test_numbered_passage_copies_exact_text_including_year_context(self):
        self.payload["sources"][1]["text"] = "Film\nIn 1940 a film was released. It was directed by Terry O. Morse."
        self.extracted["people"][1]["bounds"][0]["evidence"] = [{"source": 2, "passage": 2}]
        judge = Mock(side_effect=[self.extracted, {"supported": True}, {"supported": True}])
        result = check_birth_order(None, self.payload, judge)
        self.assertEqual(result["status"], "stop")
        self.assertEqual(result["evidence"][0]["quote"], self.payload["sources"][1]["text"].splitlines()[1])

    def test_invalid_passage_cannot_supply_a_premise(self):
        self.extracted["people"][1]["bounds"][0]["evidence"] = [{"source": 2, "passage": 99}]
        result = check_birth_order(None, self.payload, Mock(return_value=self.extracted))
        self.assertEqual(result["status"], "continue")

    def test_unverified_activity_cannot_trigger_stop(self):
        judge = Mock(side_effect=[self.extracted, {"supported": False}, {"supported": True}])
        result = check_birth_order(None, self.payload, judge)
        self.assertEqual(result["status"], "continue")
        self.assertFalse(result["handled"])

    def test_later_question_returns_other_person(self):
        self.payload["question"] = self.payload["question"].replace("earlier", "later")
        judge = Mock(side_effect=[self.extracted, {"supported": True}, {"supported": True}])
        self.assertEqual(check_birth_order(None, self.payload, judge)["answer"], "Jon Paul Puno")

    def test_exact_birth_date_does_not_accept_activity_bound(self):
        self.assertFalse(birth_order_hint("What is Terry O. Morse's exact birth date?"))
        judge = Mock()
        result = check_birth_order(None, {"question": "When was Terry O. Morse born?"}, judge)
        self.assertFalse(result["handled"])
        judge.assert_not_called()

    def test_boolean_birth_question_cannot_return_a_person_name(self):
        judge = Mock()
        result = check_birth_order(None, {"question": "Were P and Q both born first in their families?"}, judge)
        self.assertFalse(result["handled"])
        judge.assert_not_called()

    def test_multi_unknown_and_requested_year_stay_with_semantic_checker(self):
        self.assertFalse(birth_order_hint("Between P and Q, who was born first and where?"))
        self.assertFalse(birth_order_hint("Which year was P born earlier than Q?"))

    def test_same_birth_year_allows_finer_date_check(self):
        self.payload["sources"][1]["text"] = "Terry O. Morse was born on January 1, 1985."
        bound = self.extracted["people"][1]["bounds"][0]
        bound.update(relation="born_in", year=1985,
            evidence=[{"source": 2, "quote": self.payload["sources"][1]["text"]}])
        result = check_birth_order(None, self.payload, Mock(return_value=self.extracted))
        self.assertFalse(result["handled"])

    def test_bc_years_never_use_positive_integer_order(self):
        for era in ["BC", "bce", "B.C.", "B.C.E."]:
            with self.subTest(era=era):
                payload = {**self.payload, "sources": [{"source": 1,
                    "text": f"Jon Paul Puno was born in 50 {era}. Terry O. Morse was born in 100 {era}."}]}
                judge = Mock()
                result = check_birth_order(None, payload, judge)
                self.assertEqual(result["status"], "continue")
                self.assertFalse(result["handled"])
                judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
