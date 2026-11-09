from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
from grepseek.trainer.verl_integration.hybrid_agent_loop import HybridAgentLoop
from verl import DataProto
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopWorker, _InternalAgentLoopOutput
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.answer_body_gate import AnswerBodyGate, HEADER, object_array
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config.actor import ActorConfig


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i+1) for i in range(len(text))]}

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def example_batch():
    # Five complete rounds, then an answer or rejected sixth request. The final
    # model tokens deliberately carry the same +/- credit as the tool requests.
    rounds = [{"round": j+1, "generated_start": 5*j, "generated_end": 5*j+3,
               "prefix_end": 5*j+5, "tool_name": "shell" if j in [1, 3] else "dense_search"}
              for j in range(5)]
    response = torch.tensor([[1, 1, 1, 0, 0] * 5 + [1, 1, 1, 1]] * 2)
    return DataProto.from_dict(tensors={
        "prompts": torch.tensor([[0, 42, 43], [0, 42, 43]]),
        "responses": torch.arange(29).repeat(2, 1),
        "attention_mask": torch.tensor([[0, 1, 1] + [1]*29]*2),
        "response_mask": response,
        "advantages": torch.tensor([[1.25]*29, [-1.25]*29]) * response,
        "returns": torch.full((2, 29), 3.5),
        "token_level_rewards": torch.full((2, 29), .25),
        "old_log_probs": torch.full((2, 29), -.2),
    }, non_tensors={"retrieval_rounds": object_array([rounds, [dict(r) for r in rounds]]),
                    "extra_info": object_array([{"qid": "a", "golden_answers": ["", "Paris", "Other"]},
                                                {"qid": "b", "golden_answers": ["Paris"]}]),
                    "uid": np.array(["u1", "u2"], dtype=object)})


class AnswerBodyGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = AnswerBodyGate(CharacterTokenizer())
        self.batch = example_batch()

    def score(self, probabilities):
        def fn(jobs):
            self.assertEqual(len(jobs), 10)
            # Ensure padding is absent, initial prompt left padding is stripped,
            # and only body[:-1] follows the forced, unscored header.
            for ids, target in zip(jobs.non_tensor_batch["gate_input_ids"], jobs.non_tensor_batch["gate_target_ids"]):
                self.assertEqual(ids[:2], [42, 43])
                self.assertEqual(ids[-len(HEADER)-len(target)+1:], list(map(ord, HEADER+"Pari")))
                self.assertEqual(target, list(map(ord, "Paris")))
            return DataProto.from_dict(tensors={"answer_body_logp": torch.tensor([math.log(x) for x in probabilities]*2, dtype=torch.float64)})
        return fn

    def test_preserve_bridge_first_answer_invalid_tail_and_all_negative_advantages(self):
        before = {k: v.clone() for k, v in self.batch.batch.items()}
        m = self.gate.apply(self.batch, self.score([1e-7, 1e-10, .999, .9991, .9992]), step=1)
        expected = before["advantages"].clone()
        expected[0, 15:18] = 0
        expected[0, 20:23] = 0
        torch.testing.assert_close(self.batch.batch["advantages"], expected, rtol=0, atol=0)
        for k, original in before.items():
            if k != "advantages":
                torch.testing.assert_close(self.batch.batch[k], original, rtol=0, atol=0)
        self.assertEqual(m["answer_body_gate/probability_eligible_rounds"], 4)
        self.assertEqual(m["answer_body_gate/gated_rounds"], 2)
        self.assertEqual(m["answer_body_gate/gated_shell_rounds"], 1)
        self.assertEqual(m["answer_body_gate/gated_positive_adv_tokens"], 6)
        self.assertEqual(m["answer_body_gate/gated_negative_adv_tokens"], 0)
        self.assertEqual(m["answer_body_gate/gated_zero_adv_tokens"], 0)
        self.assertEqual(m["answer_body_gate/preserved_negative_adv_tokens"], 6)

    def test_high_confidence_necessary_evidence_gain_is_retained(self):
        before = self.batch.batch["advantages"].clone()
        self.gate.apply(self.batch, self.score([.986937, .999970, .5, .9, .99]), step=1)
        torch.testing.assert_close(self.batch.batch["advantages"][:, 5:8], before[:, 5:8])

    def test_actual_gspo_local_gradient_zero_without_changing_other_token_gradients(self):
        config = ActorConfig(strategy="fsdp", rollout_n=5, ppo_micro_batch_size_per_gpu=1,
                             clip_ratio_low=.2, clip_ratio_high=.28)
        old = self.batch.batch["old_log_probs"]
        original = self.batch.batch["advantages"].clone()
        def gradients(advantages):
            current = old.clone().requires_grad_(True)
            loss, _ = compute_policy_loss_gspo(old, current, advantages,
                                              self.batch.batch["response_mask"], config=config)
            return torch.autograd.grad(loss, current)[0]
        before = gradients(original)
        self.gate.apply(self.batch, self.score([1e-7, 1e-10, .999, .9991, .9992]), step=1)
        after = gradients(self.batch.batch["advantages"])
        changed = original != self.batch.batch["advantages"]
        self.assertTrue(bool(changed.any()))
        self.assertFalse(bool(changed[1].any()))
        self.assertTrue(bool((before[changed] != 0).all()))
        self.assertTrue(bool((after[changed] == 0).all()))
        torch.testing.assert_close(after[~changed], before[~changed], rtol=0, atol=0)

    def test_real_severe_drop_counterexample_preserves_even_positive_credit(self):
        before = self.batch.batch["advantages"].clone()
        m = self.gate.apply(self.batch, self.score([.2, .9999757613649302, .0033328121171042643, .2, .2]), step=1)
        torch.testing.assert_close(self.batch.batch["advantages"], before, rtol=0, atol=0)
        self.assertEqual(m["answer_body_gate/excluded_probability_drop_rounds"], 2)
        self.assertEqual(m["answer_body_gate/gated_rounds"], 0)

    def test_relative_drop_boundary_and_small_decline(self):
        for ratio, expected_rounds in [(1-.0049, 1), (1-.005, 1), (1-.0051, 0)]:
            with self.subTest(ratio=ratio):
                self.batch = example_batch()
                before = self.batch.batch["advantages"].clone()
                m = self.gate.apply(self.batch, self.score([1., ratio, .2, .2, .2]), step=1)
                self.assertEqual(m["answer_body_gate/gated_rounds"], expected_rounds)
                torch.testing.assert_close(self.batch.batch["advantages"][1], before[1], rtol=0, atol=0)

    def test_select_positive_tokens_within_mixed_sign_round(self):
        self.batch.batch["advantages"][0, 5:8] = torch.tensor([1.25, -.75, 0.])
        before = self.batch.batch["advantages"].clone()
        m = self.gate.apply(self.batch, self.score([.999, .999, .2, .2, .2]), step=1)
        expected = before.clone()
        expected[0, 5] = 0
        torch.testing.assert_close(self.batch.batch["advantages"], expected, rtol=0, atol=0)
        self.assertEqual(m["answer_body_gate/gated_generated_tokens"], 1)
        self.assertEqual(m["answer_body_gate/gated_rounds"], 1)
        self.assertEqual(m["answer_body_gate/preserved_negative_adv_tokens"], 4)

    def test_zero_and_negative_only_rounds_are_never_reported_as_cleared(self):
        self.batch.batch["advantages"][0] = 0
        before = self.batch.batch["advantages"].clone()
        m = self.gate.apply(self.batch, self.score([.999]*5), step=1)
        torch.testing.assert_close(self.batch.batch["advantages"], before, rtol=0, atol=0)
        self.assertEqual(m["answer_body_gate/probability_eligible_rounds"], 8)
        self.assertEqual(m["answer_body_gate/gated_rounds"], 0)
        self.assertEqual(m["answer_body_gate/gated_generated_tokens"], 0)

    def test_reject_invalid_drop_limits(self):
        for limit in [-.001, 1., float("nan"), float("inf")]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                AnswerBodyGate(CharacterTokenizer(), max_relative_probability_drop=limit)

    def test_large_probability_drop_requires_high_p1(self):
        before = self.batch.batch["advantages"].clone()
        self.gate.apply(self.batch, self.score([.8, .1, .01, .001, .0001]), step=1)
        torch.testing.assert_close(self.batch.batch["advantages"], before)

    def test_no_adjacent_rounds_does_not_call_actor(self):
        for i in range(2):
            self.batch.non_tensor_batch["retrieval_rounds"][i] = self.batch.non_tensor_batch["retrieval_rounds"][i][:1]
        def forbidden(_):
            self.fail("Single-round trajectory must not launch a probe")
        self.gate.apply(self.batch, forbidden, step=1)

    def test_nonfinite_probe_keeps_adjacent_credit(self):
        before = self.batch.batch["advantages"].clone()
        def scores(_):
            return DataProto.from_dict(tensors={"answer_body_logp": torch.tensor([float("nan")]*10)})
        m = self.gate.apply(self.batch, scores, step=1)
        torch.testing.assert_close(self.batch.batch["advantages"], before)
        self.assertEqual(m["answer_body_gate/skipped_prefixes"], 10)

    def test_context_overflow_keeps_credit(self):
        self.gate.max_model_len = 2
        before = self.batch.batch["advantages"].clone()
        def forbidden(_):
            self.fail("Over-context prefix must not be sent to the model")
        m = self.gate.apply(self.batch, forbidden, step=1)
        torch.testing.assert_close(self.batch.batch["advantages"], before)
        self.assertEqual(m["answer_body_gate/skipped_prefixes"], 10)

    def test_reject_metadata_that_would_clear_observation(self):
        self.batch.non_tensor_batch["retrieval_rounds"][0][2]["generated_end"] = 14
        with self.assertRaisesRegex(RuntimeError, "boundaries"):
            self.gate.apply(self.batch, lambda x: None, step=1)

    def test_same_gold_alias_and_body_only_targets(self):
        header, body, answer = self.gate.target([" ", "Paris", "Paris, France"])
        self.assertEqual(answer, "Paris")
        self.assertEqual(header, list(map(ord, HEADER)))
        self.assertEqual(body, list(map(ord, "Paris")))

    def test_dp_padding_keeps_job_order_and_variable_length_objects(self):
        jobs = DataProto.from_dict(tensors={"gate_job_id": torch.arange(10)}, non_tensors={
            "gate_input_ids": object_array([list(range(i+1)) for i in range(10)])})
        padded, n = pad_dataproto_to_divisor(jobs, 8)
        self.assertEqual(len(padded), 16)
        restored = unpad_dataproto(DataProto.concat(padded.chunk(8)), n)
        self.assertEqual(restored.batch["gate_job_id"].tolist(), list(range(10)))
        self.assertEqual(restored.non_tensor_batch["gate_input_ids"].tolist(), jobs.non_tensor_batch["gate_input_ids"].tolist())


class RetrievalRoundMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_only_complete_round_and_actual_model_span(self):
        agent = object.__new__(HybridAgentLoop)
        agent.record_retrieval_rounds = True
        data = AgentData([], None, None, {}, "test", {})
        data.prompt_ids = [99]
        data.extra_fields["retrieval_rounds"] = []
        async def generate(_, d, params, ignore=False):
            d.response_ids = [4, 5, 6]
            d.prompt_ids += d.response_ids
            d.response_mask += [1] * len(d.response_ids)
            d.tool_calls = [FunctionCall(name="shell", arguments="{}")]
            return AgentState.PROCESSING_TOOLS
        async def tools(_, d):
            d.prompt_ids += [8, 9]
            d.response_mask += [0, 0]
            d.user_turns += 1
            return AgentState.GENERATING
        with patch.object(GrepSeekAgentLoop, "_handle_generating_state", generate), patch.object(GrepSeekAgentLoop, "_handle_processing_tools_state", tools):
            await agent._handle_generating_state(data, {})
            await agent._handle_processing_tools_state(data)
        self.assertEqual(data.extra_fields["retrieval_rounds"], [{"generated_start": 0, "generated_end": 3, "prefix_end": 5, "round": 1, "tool_name": "shell"}])
        async def no_observation(_, d):
            return AgentState.TERMINATED
        with patch.object(GrepSeekAgentLoop, "_handle_generating_state", generate), patch.object(GrepSeekAgentLoop, "_handle_processing_tools_state", no_observation):
            await agent._handle_generating_state(data, {})
            await agent._handle_processing_tools_state(data)
        self.assertEqual(len(data.extra_fields["retrieval_rounds"]), 1)

    def terminal_fixture(self, *, context_limit=False, assistant_limit=False):
        agent = object.__new__(HybridAgentLoop)
        agent.record_retrieval_rounds = True
        agent.tokenizer = CharacterTokenizer()
        agent.loop = asyncio.get_running_loop()
        agent.tool_limit_behavior = "truncate_invalid"
        agent.force_final_answer_at_tool_limit = False
        agent.prefill_final_answer_at_tool_limit = False
        agent.prompt_length, agent.response_length = 1024, 100
        agent.max_user_turns, agent.max_assistant_turns = 5, 6
        agent.max_parallel_calls = 1
        agent.interaction_config_file = None
        agent.processor = None
        agent.tool_parser_name = "hermes"
        agent.tools = {"shell": SimpleNamespace(tool_schema={"name": "shell"})}
        agent.tool_parser = SimpleNamespace(extract_tool_calls=AsyncMock(return_value=("", [])))
        data = AgentData([], None, None, {}, "terminal-regression", {})
        data.prompt_ids = [99, 4, 5, 6, 8, 9]
        data.response_mask = [1, 1, 1, 0, 0]
        data.response_ids = [4, 5, 6]
        # The real parent leaves the previous tool_calls untouched when an
        # early termination condition fires before tool parsing.
        data.tool_calls = [FunctionCall(name="shell", arguments="{}")]
        data.user_turns = data.assistant_turns = 1
        data.extra_fields["retrieval_rounds"] = [{"round": 1, "generated_start": 0,
            "generated_end": 3, "prefix_end": 5, "tool_name": "shell"}]
        generated = agent.tokenizer.encode("Thinking")
        if context_limit:
            agent.response_length = len(data.response_mask) + len(generated)
        if assistant_limit:
            data.assistant_turns = agent.max_assistant_turns - 1
        agent.server_manager = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            token_ids=generated, log_probs=[-.1] * len(generated), extra_fields={},
            num_preempted=0, routed_experts=None)))
        return agent, data, generated

    async def test_terminal_context_budget_does_not_export_pending_state(self):
        agent, data, generated = self.terminal_fixture(context_limit=True)
        before = list(data.prompt_ids)
        state = await agent._handle_generating_state(data, {"temperature": 1.0})
        self.assertEqual(state, AgentState.TERMINATED)
        agent.tool_parser.extract_tool_calls.assert_not_awaited()
        self.assertTrue(data.tool_calls)  # Exercise the actual stale-call path.
        self.assertEqual(data.prompt_ids, before + generated)
        self.assertEqual(data.response_mask, [1, 1, 1, 0, 0] + [1] * len(generated))
        self.assertEqual(len(data.extra_fields["retrieval_rounds"]), 1)
        self.assertNotIn("_pending_retrieval_round", data.extra_fields)

    async def test_terminal_assistant_budget_does_not_export_pending_state(self):
        agent, data, _ = self.terminal_fixture(assistant_limit=True)
        state = await agent._handle_generating_state(data, {"temperature": 1.0})
        self.assertEqual(state, AgentState.TERMINATED)
        agent.tool_parser.extract_tool_calls.assert_not_awaited()
        self.assertTrue(data.tool_calls)
        self.assertNotIn("_pending_retrieval_round", data.extra_fields)

    async def test_real_tool_response_overflow_is_not_a_completed_round(self):
        agent, data, _ = self.terminal_fixture()
        data.response_ids = [4, 5, 6]
        data.prompt_ids = [99] + data.response_ids
        data.response_mask = [1, 1, 1]
        data.extra_fields["retrieval_rounds"] = []
        data.user_turns = 0
        agent.response_length = 5
        agent.apply_chat_template = AsyncMock(return_value=[8, 9])
        with patch.object(HybridAgentLoop, "_call_tool", new_callable=AsyncMock,
                          return_value=(ToolResponse(text="observation"), 0.0, {})) as execute:
            state = await agent._handle_processing_tools_state(data)
        execute.assert_awaited_once()
        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(data.response_mask, [1, 1, 1])
        self.assertEqual(data.extra_fields["retrieval_rounds"], [])
        self.assertNotIn("_pending_retrieval_round", data.extra_fields)

    @staticmethod
    def export_gate_metadata(data):
        # Use production worker postprocessing, with identical non-gate schema.
        # This isolates the optional gate column that crashed worker merging.
        response = data.prompt_ids[1:]
        width = 32
        assert len(response) <= width
        padded = response + [0] * (width - len(response))
        return _InternalAgentLoopOutput(
            prompt_ids=torch.tensor([[99]]), response_ids=torch.tensor([padded]),
            response_mask=torch.tensor([data.response_mask + [0] * (width-len(response))]),
            input_ids=torch.tensor([[99] + padded]),
            attention_mask=torch.tensor([[1] * (1+len(response)) + [0] * (width-len(response))]),
            position_ids=torch.arange(width+1).unsqueeze(0),
            num_turns=data.user_turns + data.assistant_turns + 1,
            metrics=AgentLoopMetrics(), multi_modal_inputs={},
            extra_fields={k: v for k, v in data.extra_fields.items()
                          if k == "retrieval_rounds" or k == "_pending_retrieval_round"})

    async def test_two_worker_640_plus640_merge_with_terminal_and_no_tool_rows(self):
        agent, terminal, _ = self.terminal_fixture(context_limit=True)
        await agent._handle_generating_state(terminal, {"temperature": 1.0})
        _, complete, _ = self.terminal_fixture()
        no_tools = AgentData([], None, None, {}, "no-tool", {})
        no_tools.prompt_ids, no_tools.response_mask = [99, 7], [1]
        no_tools.extra_fields["retrieval_rounds"] = []
        worker = object.__new__(AgentLoopWorker)
        worker.reward_loop_worker_handles = None
        a = worker._postprocess([self.export_gate_metadata(terminal)] + [self.export_gate_metadata(no_tools)] * 639)
        b = worker._postprocess([self.export_gate_metadata(complete)] * 640)
        # Before the fix this is the exact production AssertionError:
        # key _pending_retrieval_round length640 != batch size1280.
        merged = DataProto.concat([a, b])
        self.assertEqual(len(merged), 1280)
        self.assertNotIn("_pending_retrieval_round", merged.non_tensor_batch)
        self.assertTrue(all(len(v) == 1280 for v in merged.non_tensor_batch.values()))
        self.assertEqual(merged.non_tensor_batch["retrieval_rounds"][1], [])
        self.assertEqual(merged.non_tensor_batch["retrieval_rounds"][640], complete.extra_fields["retrieval_rounds"])


if __name__ == "__main__":
    unittest.main()
