from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]

from utils.branching import ExecutionBudget, state_digest
from utils.hybrid_pipeline import answer_equivalence, audit_provenance, propose_step, unseen_numbers


class BranchingTests(unittest.TestCase):
    def test_shared_state_digest_ignores_random_handles_and_latency(self):
        def steps(handle, elapsed):
            return [{"think": "Search Ada.", "action": {"name": "dense_search", "arguments": {"query": "Ada"}},
                     "observation": {"candidate_set_id": handle, "documents": [{"contents": "Ada"}]},
                     "metrics": {"elapsed_seconds": elapsed}}]
        self.assertEqual(state_digest("Ada?", steps("first", 1)), state_digest("Ada?", steps("second", 2)))

    def test_execution_budget_has_reserved_capacity(self):
        budget = ExecutionBudget(3)
        budget.active_limit = 2
        self.assertTrue(budget.consume())
        self.assertTrue(budget.consume())
        self.assertFalse(budget.consume())
        budget.active_limit = 3
        self.assertTrue(budget.consume())
        self.assertFalse(budget.consume())

    def test_future_provenance_is_rejected(self):
        llm = Mock()
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"verdict": "PASS", "complete": True,
            "mentions": [{"turn": 1, "text": "London", "source_turn": 1, "quote": "London"}]}))
        record = {"question": "Where was Ada born?", "steps": [{"think": "London?", "action": {
            "name": "dense_search", "arguments": {"query": "Ada birthplace"}}, "observation": {
            "tool": "dense_search", "status": "ok", "documents": [{"contents": "London"}]}}],
            "final": {"think": "London."}}
        _, failure = audit_provenance(llm, record)
        self.assertEqual(failure, "invalid_provenance_chronology")

    def test_rejected_proposal_is_not_used_in_retry_history(self):
        llm = Mock()
        llm.model = "qwen3.8-27b"
        llm.chat.side_effect = [SimpleNamespace(text=json.dumps({"reasoning": "I remember London.",
            "action": {"name": "dense_search", "arguments": {"query": "Ada birthplace"}}, "answer": None})),
            SimpleNamespace(text=json.dumps({"reasoning": "Find Ada birthplace.", "action": {
            "name": "dense_search", "arguments": {"query": "Ada birthplace"}}, "answer": None}))]
        record = {"gold_answers": ["London"], "model_responses": []}
        result = propose_step(llm, [{"role": "user", "content": "Where was Ada born?"}], record,
            "Where was Ada born?", "dense_search", 0, 5, 0.7, 100, 1, 1, True, None)
        self.assertIsNone(result[-1])
        self.assertNotIn("London", json.dumps(llm.chat.call_args_list[1].args[0]))
        self.assertTrue(unseen_numbers("In 1776", "Boston"))
        self.assertFalse(unseen_numbers("In 1776", "Boston 1776"))

    def test_answer_alias_requires_literal_support_and_rejects_year_mismatch(self):
        llm = Mock()
        record = {"question": "Which ocean borders New Hampshire?", "gold_answers": ["the Atlantic"],
                  "final": {"answer": "Atlantic Ocean"}, "steps": [{"observation": {
                      "tool": "dense_search", "status": "ok", "documents": [{
                          "contents": "New Hampshire borders the Atlantic Ocean."}]}}]}
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"equivalent": True,
            "evidence": [{"turn": 1, "quote": "New Hampshire borders the Atlantic Ocean."}]}))
        self.assertTrue(answer_equivalence(llm, record)[1])
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"equivalent": True,
            "evidence": [{"turn": 1, "quote": "An unsupported quote."}]}))
        self.assertFalse(answer_equivalence(llm, record)[1])
        record["gold_answers"] = ["1869"]
        record["final"]["answer"] = "March 29, 1868"
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"equivalent": True,
            "evidence": [{"turn": 1, "quote": "New Hampshire borders the Atlantic Ocean."}]}))
        self.assertFalse(answer_equivalence(llm, record)[1])


if __name__ == "__main__":
    unittest.main()
