from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]

from utils.branching import select_states
from utils.branching import run_question
from utils.hybrid_pipeline import _phrase_supported, audit_provenance, generate_trajectory, leakage_failure, propose_step, sft_record, visible_text
from utils.readiness import audit_prefixes, check_ready, prefix_payload
from utils.answer_grounding import answer_support
from utils.conclusion import check_conclusion, conclusion_payload
from utils.logic import boolean_hint, check_short_circuit


class StopProtocolTests(unittest.TestCase):
    def test_boolean_counterexample_stops_without_unknown_other_subject(self):
        payload = {"question": "Are Fitness and Stylist catered to men?", "sources": [{"source": 1, "text": "Stylist is a magazine for women."}]}
        atom = {"operator": "AND", "predicate": "catered to men", "claims": [{"entity": "Fitness", "truth": None},
            {"entity": "Stylist", "truth": False, "evidence": [{"source": 1, "quote": "Stylist is a magazine for women."}]}]}
        judge = Mock(side_effect=[atom, {"supported": True}])
        self.assertEqual(check_short_circuit(None, payload, judge)["answer"], "No")
        verifier_input = json.loads(judge.call_args_list[1].args[1].split("INPUT:\n", 1)[1])
        self.assertNotIn("question", verifier_input)
        self.assertNotIn("Fitness", json.dumps(verifier_input))

    def test_one_positive_and_unknown_still_allow_needed_retrieval(self):
        payload = {"question": "Are P and Q both bands?", "sources": [{"source": 1, "text": "P is a band."}]}
        judge = Mock(return_value={"operator": "AND", "predicate": "bands", "claims": [{"entity": "P", "truth": True}, {"entity": "Q", "truth": None}]})
        self.assertEqual(check_short_circuit(None, payload, judge)["status"], "continue")
        judge.assert_called_once()

    def test_no_hit_cannot_be_fabricated_as_negative_premise(self):
        payload = {"question": "Are P and Q both bands?", "sources": []}
        judge = Mock(return_value={"operator": "AND", "predicate": "bands", "claims": [{"entity": "P", "truth": None},
            {"entity": "Q", "truth": False, "evidence": [{"source": 1, "quote": "Q is not a band."}]}]})
        self.assertEqual(check_short_circuit(None, payload, judge)["status"], "uncertain")

    def test_atomic_predicate_cannot_be_swapped(self):
        payload = {"question": "Are P and Q catered to men?", "sources": []}
        judge = Mock(return_value={"operator": "AND", "predicate": "catered to women", "claims": [{}, {}]})
        self.assertEqual(check_short_circuit(None, payload, judge)["status"], "uncertain")

    def test_boolean_hint_excludes_relational_comparisons(self):
        self.assertFalse(boolean_hint("Are P and Q in the same state?"))
        self.assertFalse(boolean_hint("Does P or Q have more species?"))
        self.assertTrue(boolean_hint("Did P and Q both have more than one act?"))

    def test_boolean_numeric_threshold_is_not_a_relational_comparison(self):
        self.assertTrue(boolean_hint("Are P and Q both operas with more than one act?"))
        self.assertTrue(boolean_hint("Do P and Q both have more than 5 members?"))
        self.assertFalse(boolean_hint("Do P and Q have more species than R?"))

    def test_provenance_accepts_generic_phone_synonym_without_licensing_new_entity(self):
        llm = Mock(model="qwen")
        record = {"question": "Which person suggested hello for the phone?", "steps": [{
            "think": "Search telephone greeting.", "action": {"name": "dense_search", "arguments": {"query": "telephone greeting"}},
            "observation": {"tool": "dense_search", "status": "ok", "documents": [{"contents": "Edison suggested hello."}]}}],
            "final": {"think": "Edison suggested hello."}}
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"verdict": "PASS", "complete": True,
            "mentions": [{"turn": 1, "text": "telephone", "source_turn": 0, "quote": record["question"]}]}))
        self.assertIsNone(audit_provenance(llm, record)[1])
        self.assertTrue(_phrase_supported("United States", "U.S."))
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"verdict": "PASS", "complete": True,
            "mentions": [{"turn": 1, "text": "Edison", "source_turn": 0, "quote": record["question"]}]}))
        record["steps"][0]["think"] = "Edison suggested it."
        self.assertEqual(audit_provenance(llm, record)[1], "invalid_provenance_source")

    def test_final_provenance_does_not_accept_name_from_unnamed_definition(self):
        llm = Mock(model="qwen")
        question = "Which conflict was fought by China and Japan from 1937 to 1945?"
        record = {"question": question, "steps": [], "final": {"think": "Second Sino-Japanese War."}}
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"verdict": "PASS", "complete": True,
            "mentions": [{"turn": 1, "text": "Second Sino-Japanese War", "source_turn": 0, "quote": question}]}))
        self.assertEqual(audit_provenance(llm, record)[1], "invalid_provenance_source")

    def test_definition_does_not_license_unseen_named_answer(self):
        result = answer_support("Which war was fought by China and Japan from 1937 to 1945?", [], "Second Sino-Japanese War")
        self.assertFalse(result["supported"])

    def test_direct_question_and_country_surface_alias_are_allowed(self):
        self.assertTrue(answer_support("Bishkek is in Kyrgyzstan. Which country?", [], "Kyrgyzstan")["supported"])
        self.assertTrue(answer_support("The U.S. state of New Jersey. Which country?", [], "United States")["supported"])
        self.assertFalse(answer_support("New South Wales. Which country?", [], "Australia")["supported"])

    def test_bridge_entity_is_not_the_requested_property(self):
        self.assertFalse(answer_support("Where was the author of Novel born?", ["Novel was written by Ada."], "London")["supported"])
        self.assertTrue(answer_support("Where was the author of Novel born?", ["Novel was written by Ada.", "Ada was born in London."], "London")["supported"])

    def test_answers_can_compose_visible_names_without_copying_a_full_sentence(self):
        self.assertTrue(answer_support("Who won?", ["Harold Godwinson led the English army. It defeated the invaders at Stamford Bridge."],
            "Harold Godwinson (the English army) won at Stamford Bridge.")["supported"])
        self.assertFalse(answer_support("Which war?", ["China and Japan fought from 1937 to 1945."], "Second Sino-Japanese War")["supported"])

    def test_forced_tool_cannot_veto_answer(self):
        llm = Mock(model="qwen")
        llm.chat.return_value = SimpleNamespace(text=json.dumps({
            "reasoning": "The passage states London.", "action": None, "answer": "London"}))
        record = {"gold_answers": ["London"], "model_responses": []}
        result = propose_step(llm, [], record, "Ada was born in London.", "dense_search",
                              1, 5, .7, 100, 1, 2, True, None)
        self.assertEqual(result[2], "London")
        self.assertIsNone(result[3])
        self.assertNotIn("proposal_rejections", record)

    def test_derived_final_is_allowed_only_after_semantic_audits(self):
        record = {"question": "The company is American. In which country?", "steps": [],
                  "gold_answers": ["United States"], "final": {"think": "American identifies the United States."}}
        self.assertEqual(leakage_failure(record), "unseen_gold_in_reasoning")
        record.update(stop_protocol="prefix_v1", provenance_check={"verdict": "PASS"},
                      quality_check={"verdict": "PASS"}, stop_audit_passed=True)
        self.assertIsNone(leakage_failure(record))
        record["stop_audit_passed"] = False
        self.assertEqual(leakage_failure(record), "unseen_gold_in_reasoning")

    def test_controller_stops_before_forced_first_tool_with_question_evidence(self):
        llm = Mock(model="qwen")
        llm.chat.return_value = SimpleNamespace(text=json.dumps({
            "reasoning": "Find the country of Bishkek.", "answer": None,
            "action": {"name": "dense_search", "arguments": {"query": "Bishkek country"}}}))
        environment = Mock()
        environment.tokenizer.apply_chat_template.return_value = list(range(20))
        question = "Bishkek is in Kyrgyzstan. Which country contains Bishkek?"
        certificate = {"status": "ready", "answer": "Kyrgyzstan", "reason": "The question places Bishkek in Kyrgyzstan."}
        with patch("utils.hybrid_pipeline.check_ready", return_value=certificate), patch(
            "utils.hybrid_pipeline.check_conclusion", return_value={"state": "incomplete", "reason": "Needs country."}), patch(
            "utils.hybrid_pipeline.judge_call", return_value={"verdict": "FAIL"}):
            record = generate_trajectory(llm, environment, {"id": "q", "question": question,
                "golden_answers": ["Kyrgyzstan"]}, "system", "q::dense_first", "dense_first",
                stop_when_ready=True, check_redundancy=False)
        environment.execute.assert_not_called()
        self.assertEqual(record["steps"], [])
        self.assertEqual(record["final"]["answer"], "Kyrgyzstan")
        self.assertNotEqual(record["abort_reason"], "no_retrieved_evidence")

    def test_prefix_requests_exclude_future_gold_and_later_reasoning(self):
        steps = [{"think": "Search Ada birthplace.", "observation": {
            "status": "ok", "tool": "dense_search", "documents": [{"contents": "Ada was born in London."}]}},
            {"think": "The answer is London; confirm it.", "observation": {
                "status": "ok", "tool": "dense_search", "documents": [{"contents": "FUTURE_SECRET"}]}}]
        record = {"question": "Where was Ada born?", "steps": steps, "gold_answers": ["GOLD_SECRET"],
                  "conclusion_checks": [{"before_tool": i, "state": "incomplete", "reason": "Need birthplace."} for i in (1, 2)]}
        judge = Mock(side_effect=[{"redundant": False, "missing_relation": "birthplace"},
            {"redundant": True, "reasoning_quote": "The answer is London", "evidence": []}])
        checks, failure = audit_prefixes(None, record, judge, visible_text)
        self.assertEqual(failure, "already_answerable_before_tool")
        first, second = [call.args[1] for call in judge.call_args_list]
        self.assertNotIn("London", first)
        self.assertNotIn("FUTURE_SECRET", second)
        self.assertNotIn("GOLD_SECRET", second)
        self.assertEqual(len(checks), 2)

    def test_bridge_missing_relation_can_continue(self):
        payload = {"question": "Where was the author of Novel born?", "sources": [
            {"source": 1, "text": "Novel was written by Ada."}], "proposed_reasoning": "Find Ada's birthplace."}
        judge = Mock(return_value={"status": "missing", "missing_relation": "Ada's birthplace"})
        self.assertEqual(check_ready(None, payload, judge)["status"], "missing")
        judge.assert_called_once()

    def test_invalid_quote_and_unverified_certificate_are_quarantined(self):
        payload = {"question": "Ada?", "sources": [], "proposed_reasoning": "Search."}
        judge = Mock(return_value={"status": "ready", "answer": "London", "reason": "Known.",
            "evidence": [{"source": 1, "quote": "London"}]})
        self.assertEqual(check_ready(None, payload, judge)["status"], "uncertain")
        payload["question"] = "Ada is in London."
        judge.side_effect = [{"status": "ready", "answer": "London", "reason": "Question states it.",
            "evidence": [{"source": 0, "quote": "Ada is in London."}]}, {"supported": False}]
        self.assertEqual(check_ready(None, payload, judge)["status"], "uncertain")

    def test_answered_prefix_is_not_forked(self):
        step = {"think": "Search Ada.", "action": {"name": "dense_search", "arguments": {"query": "Ada"}},
            "observation": {"status": "ok", "documents": []}, "metrics": {}}
        record = {"question": "Ada?", "steps": [step], "success": True}
        self.assertEqual(select_states([record], 5), [])

    def test_export_requires_completed_audit(self):
        record = {"success": True, "question": "Q?", "steps": [], "gold_answers": [], "stop_protocol": "prefix_v1"}
        self.assertIsNone(sft_record(record))

    def test_successful_compression_dominates_parent_before_path_diversity(self):
        import copy
        step = {"think": "Find Ada.", "action": {"name": "dense_search", "arguments": {"query": "Ada"}},
                "observation": {"status": "ok", "documents": [{"contents": "Ada was born in London."}]}, "metrics": {}}
        base = {"question": "Where was Ada born?", "success": True, "answer_matches": True,
                "abort_reason": "", "mixed": False, "final": {"answer": "London"}}
        short = {**base, "trajectory_id": "short", "steps": [step], "path": "dense_search"}
        long = {**base, "trajectory_id": "long", "steps": [step, copy.deepcopy(step)], "path": "dense_search->shell", "mixed": True}
        with patch("utils.branching.generate_trajectory", side_effect=[short, long]), patch(
            "utils.branching.judge_call", return_value={"supported": False}):
            result = run_question(None, None, {"id": "q", "question": base["question"]}, "sys",
                                  ["free", "shell_first"], max_states=0)
        self.assertEqual(result["selected_trajectories"], ["short"])
        self.assertEqual(long["abort_reason"], "dominated_by_supported_stop")

    def test_conclusion_detector_requires_a_quote_from_reasoning(self):
        payload = conclusion_payload("Which city?", [], "The city must be Z; confirm it.")
        judge = Mock(return_value={"state": "complete", "quote_turn": 1, "quote": "The city must be Z", "answer": "Z"})
        self.assertEqual(check_conclusion(None, payload, judge)["state"], "complete")
        judge.return_value = {"state": "complete", "quote_turn": 2, "quote": "Z"}
        self.assertEqual(check_conclusion(None, payload, judge)["state"], "uncertain")

    def test_conclusion_payload_cannot_see_observations_gold_or_future(self):
        payload = conclusion_payload("Q?", [{"think": "I know the author.", "observation": {"text": "SECRET"}}], "Find the birthplace.")
        serialized = json.dumps(payload)
        self.assertNotIn("SECRET", serialized)
        self.assertEqual(payload["reasoning_history"][-1]["text"], "Find the birthplace.")
        judge = Mock(return_value={"state": "incomplete", "reason": "Birthplace still missing."})
        self.assertEqual(check_conclusion(None, payload, judge)["state"], "incomplete")

    def test_complete_conclusion_never_searches_to_patch_missing_support(self):
        llm = Mock(model="qwen")
        llm.chat.return_value = SimpleNamespace(text=json.dumps({"reasoning": "Z is the described platform; verify its location.",
            "action": {"name": "dense_search", "arguments": {"query": "Z location"}}, "answer": None}))
        environment = Mock()
        with patch("utils.hybrid_pipeline.check_conclusion", return_value={"state": "complete", "quote": "Z is the described platform"}), patch(
            "utils.hybrid_pipeline.check_ready", return_value={"status": "missing", "missing_relation": "Z location"}):
            record = generate_trajectory(llm, environment, {"id": "q", "question": "Which platform?", "golden_answers": ["Z"]},
                "system", "q", stop_when_ready=True)
        environment.execute.assert_not_called()
        self.assertEqual(record["abort_reason"], "answer_concluded_without_support")

    def test_new_protocol_preserves_bridge_without_orderless_observation_deletion(self):
        llm = Mock(model="qwen")
        llm.chat.side_effect = [SimpleNamespace(text=json.dumps(v)) for v in [
            {"reasoning": "Find the author of Novel.", "action": {"name": "dense_search", "arguments": {"query": "Novel author"}}, "answer": None},
            {"reasoning": "Find Ada's birthplace.", "action": {"name": "shell", "arguments": {"command": "rg -F 'Ada' corpus.jsonl | head -n 3"}}, "answer": None},
            {"reasoning": "Ada was born in London.", "action": None, "answer": "London"}]]
        environment = Mock()
        environment.tokenizer.apply_chat_template.return_value = list(range(20))
        environment.execute.side_effect = [({"status": "ok", "tool": "dense_search", "documents": [{"contents": "Novel was written by Ada."}]}, {}),
            ({"status": "ok", "tool": "shell", "stdout": "Ada was born in London."}, {})]
        review = {"verdict": "PASS", "answer_correct": True, "grounded": True, "complete_evidence": True,
            "turns": [{"turn": 1, "contribution": "evidence"}, {"turn": 2, "contribution": "evidence"}],
            "claims": [{"id": "c1", "claim": "author", "depends_on": [], "evidence": [{"turn": 1, "quote": "Novel was written by Ada."}]},
                       {"id": "c2", "claim": "birthplace", "depends_on": ["c1"], "evidence": [{"turn": 2, "quote": "Ada was born in London."}]}]}
        with patch("utils.hybrid_pipeline.check_conclusion", return_value={"state": "incomplete", "reason": "Need next relation."}), patch(
            "utils.hybrid_pipeline.check_ready", return_value={"status": "missing", "missing_relation": "next relation"}), patch(
            "utils.hybrid_pipeline.judge_call", return_value=review) as judge, patch(
            "utils.hybrid_pipeline.audit_prefixes", return_value=([{"before_tool": 1}, {"before_tool": 2}], None)):
            record = generate_trajectory(llm, environment, {"id": "q", "question": "Where was the author of Novel born?", "golden_answers": ["London"]},
                "system", "q", stop_when_ready=True)
        self.assertTrue(record["success"])
        self.assertEqual(record["path"], "dense_search->shell")
        self.assertEqual(judge.call_count, 1)  # Quality review, no orderless deletion.


if __name__ == "__main__":
    unittest.main()
