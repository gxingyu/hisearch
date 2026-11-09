from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]

from utils.hybrid_pipeline import _phrase_supported, generate_trajectory, leakage_failure, parse_action, sequence_length, sft_record, validate_review, visible_text


class HybridPipelineTests(unittest.TestCase):
    def test_exactly_one_action_and_reasoning(self):
        reasoning, action, answer = parse_action(
            'I need a passage.\n<tool_call>{"name":"dense_search","arguments":{"query":"Ada birthplace"}}</tool_call>'
        )
        self.assertIsNone(answer)
        self.assertEqual(action["name"], "dense_search")
        self.assertTrue(reasoning)
        with self.assertRaises(ValueError):
            parse_action('No search needed.\n<answer>Ada</answer><answer>Ada</answer>')
        with self.assertRaises(ValueError):
            parse_action('<answer>Ada</answer>')

    def test_candidate_schema_requires_handle(self):
        with self.assertRaises(ValueError):
            parse_action('Read it.\n<tool_call>{"name":"shell","arguments":{"command":"cat corpus.jsonl","scope":{"type":"candidates"}}}</tool_call>')

    def test_evidence_must_quote_visible_contents(self):
        record = {"steps": [{"observation": {
            "tool": "dense_search", "status": "ok", "documents": [
                {"title": "Hidden answer", "contents": "Ada was born in London.", "content_truncated": True}
            ],
        }}]}
        review = {
            "verdict": "PASS", "answer_correct": True, "grounded": True,
            "complete_evidence": True,
            "turns": [{"turn": 1, "contribution": "evidence"}],
            "claims": [{"id": "c1", "claim": "birthplace", "depends_on": [],
                        "evidence": [{"turn": 1, "quote": "Ada was born in London."}]}],
        }
        self.assertIsNone(validate_review(review, record))
        hidden = copy.deepcopy(review)
        hidden["claims"][0]["evidence"][0]["quote"] = "Hidden answer"
        self.assertEqual(validate_review(hidden, record), "quote_not_in_visible_evidence")
        redundant = copy.deepcopy(review)
        redundant["turns"][0]["contribution"] = "redundant"
        self.assertEqual(validate_review(redundant, record), "redundant_or_unreviewed_turn")

    def test_shell_jsonl_quotes_use_decoded_contents(self):
        observation = {"tool": "shell", "stdout": json.dumps({"id": "0", "contents": '"Ada"\nAda was born in London.'})}
        self.assertIn("Ada was born in London.", visible_text(observation))

    def test_provenance_phrase_normalizes_punctuation(self):
        self.assertTrue(_phrase_supported("its a great day", "It's a Great Day"))
        self.assertFalse(_phrase_supported("Great Day", "A different song"))

    def test_mixed_shell_output_preserves_visible_non_json_lines(self):
        observation = {"tool": "shell", "stdout": 'Visible relation.\n' + json.dumps({"contents": "Another relation."})}
        self.assertEqual(visible_text(observation), "Visible relation.\nAnother relation.")

    def test_sequence_length_supports_tokenizer_mapping_and_list(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {"input_ids": [101, 102, 103], "attention_mask": [1, 1, 1]}
        self.assertEqual(sequence_length(tokenizer, []), 3)
        tokenizer.apply_chat_template.return_value = [101, 102, 103, 104]
        self.assertEqual(sequence_length(tokenizer, []), 4)

    def test_policy_isolation_and_sequence_budget(self):
        environment = Mock()
        environment.execute.return_value = (
            {"tool": "dense_search", "status": "ok", "documents": [{"contents": "Ada was born in London."}]},
            {"elapsed_seconds": 0},
        )
        environment.tokenizer.apply_chat_template.return_value = {
            "input_ids": list(range(20)), "attention_mask": [1] * 20,
        }
        llm = Mock()
        llm.chat.side_effect = [
            SimpleNamespace(text=json.dumps({"reasoning": "Search Ada birthplace.", "action": {
                "name": "dense_search", "arguments": {"query": "Ada birthplace"}}, "answer": None})),
            SimpleNamespace(text=json.dumps({"reasoning": "The passage states London.", "action": None, "answer": "London"})),
        ]
        record = generate_trajectory(
            llm, environment, {"id": "q1", "question": "Where was Ada born?", "golden_answers": ["London"]},
            "General retrieval instructions.", "q1::dense_first::0", strategy="dense_first",
            max_calls=1, max_sequence_tokens=10, stop_when_ready=False,
        )
        self.assertEqual(record["abort_reason"], "sequence_too_long")
        self.assertEqual(record["sequence_tokens"], 20)
        self.assertEqual(llm.chat.call_count, 2)
        first_prompt = llm.chat.call_args_list[0].args[0]
        self.assertIn("Sampling instruction:", first_prompt[0]["content"])
        self.assertNotIn("London", json.dumps(first_prompt))
        self.assertEqual(record["messages"][0]["content"], "General retrieval instructions.")
        self.assertNotIn("Sampling instruction:", json.dumps(record["messages"]))
        self.assertNotIn("Retrieval budget is exhausted", json.dumps(record["messages"]))
        self.assertIn("Ada was born in London.", record["messages"][3]["content"])
        environment.session.return_value.close.assert_called_once()

    def test_unseen_answer_in_reasoning_is_rejected_before_execution(self):
        environment = Mock()
        llm = Mock()
        llm.chat.return_value = SimpleNamespace(text=json.dumps({
            "reasoning": "I recall London, but must verify it.", "answer": None,
            "action": {"name": "dense_search", "arguments": {"query": "Ada birthplace"}},
        }))
        record = generate_trajectory(
            llm, environment, {"id": "q1", "question": "Where was Ada born?", "golden_answers": ["London"]},
            "General retrieval instructions.", "q1::free::0",
        )
        self.assertEqual(record["abort_reason"], "unseen_gold_in_reasoning")
        environment.execute.assert_not_called()
        environment.session.return_value.close.assert_called_once()

    def test_audit_respects_evidence_chronology(self):
        record = {"question": "Where was Ada born?", "gold_answers": ["London"], "steps": [{
            "think": "Search for Ada birthplace.",
            "action": {"name": "dense_search", "arguments": {"query": "Ada birthplace"}},
            "observation": {"tool": "dense_search", "documents": [{"contents": "Ada was born in London."}]},
        }], "final": {"think": "The evidence states London.", "answer": "London"}}
        self.assertIsNone(leakage_failure(record))
        record["steps"][0]["think"] = "I think London; search to confirm."
        self.assertEqual(leakage_failure(record), "unseen_gold_in_reasoning")
        record["success"] = True
        self.assertIsNone(sft_record(record))


if __name__ == "__main__":
    unittest.main()
