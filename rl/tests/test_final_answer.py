from __future__ import annotations

import asyncio
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from grepseek.trainer.verl_integration.agent_loop import (
    GrepSeekAgentLoop,
    _FINAL_ANSWER_CONTENT_REGEX,
    _FINAL_ANSWER_PREFIX,
)
from grepseek.trainer.verl_integration.reward_function import _is_valid_sequence
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class FinalAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tokenizer = CharacterTokenizer()
        self.agent = object.__new__(GrepSeekAgentLoop)
        self.agent.tokenizer = self.tokenizer
        self.agent.loop = asyncio.get_running_loop()
        self.agent.tools = {}
        self.agent.prompt_length = 1024
        self.agent.response_length = 15360
        self.agent.rollout_config = SimpleNamespace(max_model_len=16384)
        self.agent.prefill_final_answer_at_tool_limit = True
        self.agent.final_answer_max_tokens = 256
        self.agent.final_answer_prefix_ids = self.tokenizer.encode(_FINAL_ANSWER_PREFIX)
        self.data = AgentData([], None, None, {}, "test", {})
        prefix = "assistant\n<think>\n"
        self.data.prompt_ids = self.tokenizer.encode(prefix)
        self.data.response_mask = [0] * len(self.data.prompt_ids)
        self.data.response_logprobs = [0.0] * len(self.data.prompt_ids)
        self.data.extra_fields["force_final_answer_next_turn"] = True

    async def test_final_answer_preserves_format_and_masks_prefill(self):
        original_params = {"temperature": 0.0, "stop": ["original-stop"]}
        original_prompt_length = len(self.data.prompt_ids)

        async def generate(agent, agent_data, sampling_params, ignore_termination=False):
            self.assertEqual(sampling_params.pop("max_tokens"), 256)
            self.assertTrue(sampling_params["include_stop_str_in_output"])
            self.assertIn("</answer>", sampling_params["stop"])
            self.assertTrue(agent.tokenizer.decode(agent_data.prompt_ids).endswith("<answer>\n"))
            agent_data.response_ids = agent.tokenizer.encode("Jonas Salk</answer>")
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            agent_data.response_logprobs += [-0.1] * len(agent_data.response_ids)
            return AgentState.TERMINATED

        with patch.object(ToolAgentLoop, "_handle_generating_state", generate):
            state = await self.agent._handle_generating_state(self.data, original_params)

        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(original_params, {"temperature": 0.0, "stop": ["original-stop"]})
        generated_start = original_prompt_length + len(self.agent.final_answer_prefix_ids)
        self.assertTrue(all(mask == 0 for mask in self.data.response_mask[:generated_start]))
        self.assertTrue(all(mask == 1 for mask in self.data.response_mask[generated_start:]))
        self.assertEqual(len(self.data.prompt_ids), len(self.data.response_mask))
        self.assertEqual(len(self.data.prompt_ids), len(self.data.response_logprobs))
        transcript = self.tokenizer.decode(self.data.prompt_ids).removeprefix("assistant\n")
        self.assertTrue(_is_valid_sequence(transcript)[0])
        self.assertFalse(self.data.extra_fields["final_answer_hit_token_limit"])

    async def test_answer_budget_respects_response_and_model_limits(self):
        prefix_length = len(self.agent.final_answer_prefix_ids)
        self.agent.response_length = len(self.data.response_mask) + prefix_length + 12
        self.agent.rollout_config.max_model_len = len(self.data.prompt_ids) + prefix_length + 8
        params = self.agent._prepare_final_answer_generation(self.data, {"temperature": 0.0})
        self.assertEqual(params["max_tokens"], 8)
        self.assertLessEqual(len(self.data.response_mask) + params["max_tokens"], self.agent.response_length)

    async def test_no_space_for_prefill_terminates_without_overflow(self):
        self.agent.response_length = len(self.data.response_mask) + len(self.agent.final_answer_prefix_ids)
        original_prompt = list(self.data.prompt_ids)
        with patch.object(ToolAgentLoop, "_handle_generating_state") as generate:
            state = await self.agent._handle_generating_state(self.data, {"temperature": 0.0})
        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(self.data.prompt_ids, original_prompt)
        self.assertTrue(self.data.extra_fields["final_answer_budget_exhausted"])
        generate.assert_not_called()

    async def test_caller_budget_is_preserved(self):
        params = self.agent._prepare_final_answer_generation(self.data, {"max_new_tokens": 16})
        self.assertEqual(params["max_tokens"], 16)
        self.assertNotIn("max_new_tokens", params)

    async def test_non_final_turn_does_not_prefill(self):
        self.data.extra_fields["force_final_answer_next_turn"] = False
        original_prompt = list(self.data.prompt_ids)

        async def generate(agent, agent_data, sampling_params, ignore_termination=False):
            self.assertEqual(agent_data.prompt_ids, original_prompt)
            self.assertNotIn("structured_outputs", sampling_params)
            agent_data.response_ids = agent.tokenizer.encode("A normal response")
            return AgentState.TERMINATED

        with patch.object(ToolAgentLoop, "_handle_generating_state", generate):
            await self.agent._handle_generating_state(self.data, {"temperature": 1.0})
        self.assertFalse(self.data.extra_fields.get("forced_final_answer", False))

    def test_answer_constraint_rejects_blank_and_unbounded_answers(self):
        self.assertIsNotNone(re.fullmatch(_FINAL_ANSWER_CONTENT_REGEX, "Jonas Salk</answer>"))
        self.assertIsNone(re.fullmatch(_FINAL_ANSWER_CONTENT_REGEX, "\n </answer>"))
        self.assertIsNone(re.fullmatch(_FINAL_ANSWER_CONTENT_REGEX, "A" * 257 + "</answer>"))


if __name__ == "__main__":
    unittest.main()
