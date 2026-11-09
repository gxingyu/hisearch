from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]
from utils.birth_order import band_order_hint, check_band_order
from utils.logic import check_short_circuit
from utils.readiness import check_ready


class BandOrderTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"question": "Which rock band was formed first, Paramore or You Am I?",
            "sources": [{"source": 1, "text": "Paramore formed in 2004."},
                        {"source": 2, "text": "The concert on 24 November 1996 had support bands including You Am I."}]}
        self.extracted = {"bands": [
            {"entity": "Paramore", "bounds": [{"relation": "formed_in", "year": 2004,
                "evidence": [{"source": 1, "passage": 1}]}]},
            {"entity": "You Am I", "bounds": [{"relation": "formed_by", "year": 1996,
                "evidence": [{"source": 2, "passage": 1}]}]}]}

    def test_past_band_activity_is_enough_without_exact_formation_date(self):
        judge = Mock(side_effect=[self.extracted, {"supported": True}, {"supported": True}])
        result = check_short_circuit(None, self.payload, judge)
        self.assertEqual((result["status"], result["answer"]), ("stop", "You Am I"))
        self.assertIn("formed by 1996", result["reason"])
        self.assertNotIn('"question"', judge.call_args_list[1].args[1].split("INPUT:\n")[1])

    def test_overlapping_activity_bound_preserves_needed_search(self):
        self.payload["sources"][1]["text"] = "You Am I performed in 2006."
        self.extracted["bands"][1]["bounds"][0]["year"] = 2006
        result = check_ready(None, self.payload, Mock(return_value=self.extracted))
        self.assertEqual(result["status"], "missing")

    def test_future_members_solo_activity_does_not_date_the_band(self):
        self.payload["sources"][1]["text"] = "A future member of You Am I performed solo in 1996."
        judge = Mock(side_effect=[self.extracted, {"supported": False}, {"supported": True}])
        self.assertNotEqual(check_band_order(None, self.payload, judge)["status"], "stop")

    def test_invented_year_cannot_be_cited(self):
        self.extracted["bands"][1]["bounds"][0]["year"] = 1989
        self.assertNotEqual(check_band_order(None, self.payload, Mock(return_value=self.extracted))["status"], "stop")

    def test_later_returns_the_other_band(self):
        self.payload["question"] = self.payload["question"].replace("first", "later")
        judge = Mock(side_effect=[self.extracted, {"supported": True}, {"supported": True}])
        self.assertEqual(check_band_order(None, self.payload, judge)["answer"], "Paramore")

    def test_unknown_other_band_cannot_be_ordered(self):
        self.extracted["bands"][1]["bounds"] = []
        self.assertEqual(check_band_order(None, self.payload, Mock(return_value=self.extracted))["status"], "continue")

    def test_rejected_atomic_bound_cannot_fall_back_to_a_whole_question_guess(self):
        self.payload["sources"][1]["text"] = "A future member of You Am I performed solo in 1996."
        judge = Mock(side_effect=[self.extracted, {"supported": False}, {"supported": True}])
        result = check_ready(None, self.payload, judge)
        self.assertEqual(result["status"], "missing")
        self.assertEqual(judge.call_count, 3)

    def test_exact_year_or_other_unknowns_never_use_activity_bound(self):
        for question in ["Which year was the band You Am I formed first?",
                "Which rock band formed first and where?", "When was You Am I formed?",
                "Which film was released first, P or Q?"]:
            with self.subTest(question=question):
                self.assertFalse(band_order_hint(question))


if __name__ == "__main__":
    unittest.main()
