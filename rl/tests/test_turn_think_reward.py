from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop, _count_turn_think_tokens
from grepseek.trainer.verl_integration.reward_function import grepseek_turn_think_reward
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class TurnThinkRewardTests(unittest.TestCase):
    def setUp(self):
        self.answer = "<think>Found the date.</think><answer>Abraham Lincoln</answer>"
        self.ground_truth = {"golden_answers": ["Abraham Lincoln"]}

    def reward(self, solution_str=None, lengths=None, **extra_info):
        return grepseek_turn_think_reward(
            data_source="test",
            solution_str=self.answer if solution_str is None else solution_str,
            ground_truth=self.ground_truth,
            extra_info={"turn_think_token_counts": [100, 600] if lengths is None else lengths, **extra_info},
        )

    def test_penalty_uses_only_excess_and_clamps(self):
        result = self.reward()
        self.assertAlmostEqual(result["score"], 1 - 0.1 * 88 / 512)
        self.assertAlmostEqual(result["reward/turn_think_penalty"], 0.1 * 88 / 512)
        self.assertEqual(result["metrics/turn_think_max_tokens"], 600)
        self.assertEqual(self.reward(lengths=[512, 512])["score"], 1.0)
        self.assertEqual(self.reward(lengths=[1500])["score"], 0.9)

    def test_missing_final_answer_gets_failed_reward(self):
        result = self.reward(solution_str="<think>Still searching", lengths=[1500], hit_max_context=True)
        self.assertEqual(result["score"], -0.2)
        self.assertEqual(result["reward/failed_rollout"], 1.0)
        self.assertEqual(result["reward/turn_think_penalty"], 0.0)
        tool_only = "<think>Found a lead.</think><tool_call>{}</tool_call>"
        self.assertEqual(self.reward(solution_str=tool_only)["score"], -0.2)

    def test_invalid_format_and_tool_request_keep_format_penalty(self):
        malformed = self.answer + "extra"
        self.assertEqual(self.reward(solution_str=malformed)["score"], -1.0)
        self.assertEqual(self.reward(solution_str="<think>Found it.</think>Abraham Lincoln")["score"], -1.0)
        self.assertEqual(
            self.reward(solution_str="<think>Still searching", invalid_tool_request=True)["score"],
            -1.0,
        )

    def test_validation_keeps_unpenalized_f1(self):
        result = self.reward(lengths=[1500], is_validation=True)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reward/turn_think_penalty"], 0.0)
        empty_answer = "<think>Unsure.</think><answer></answer>"
        self.assertEqual(self.reward(solution_str=empty_answer, is_validation=True)["score"], 0.0)

    def test_missing_lengths_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "turn_think_token_counts"):
            grepseek_turn_think_reward("test", self.answer, self.ground_truth, {})

    def test_turn_lengths_exclude_tags_and_actions(self):
        tokenizer = CharacterTokenizer()
        for output in (
            "Thinking</think><tool_call>command</tool_call>",
            "<think>Thinking</think><answer>answer</answer>",
            "Thinking<tool_call>command</tool_call>",
        ):
            self.assertEqual(_count_turn_think_tokens(tokenizer.encode(output), tokenizer), len("Thinking"))
        self.assertEqual(_count_turn_think_tokens(tokenizer.encode("<answer>answer</answer>"), tokenizer), 0)
        unfinished = "Thinking forever"
        self.assertEqual(_count_turn_think_tokens(tokenizer.encode(unfinished), tokenizer), len(unfinished))


class TurnThinkAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_records_each_turn(self):
        agent = object.__new__(GrepSeekAgentLoop)
        agent.tokenizer = CharacterTokenizer()
        agent.loop = asyncio.get_running_loop()
        agent.tools = {}
        agent.response_length = 100
        agent.prefill_final_answer_at_tool_limit = False
        data = AgentData([], None, None, {}, "test", {})
        outputs = iter(("First thought</think><answer>no</answer>", "More thought</think><answer>yes</answer>"))

        async def generate(agent_loop, agent_data, sampling_params, ignore_termination=False):
            agent_data.response_ids = agent_loop.tokenizer.encode(next(outputs))
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            return AgentState.TERMINATED

        with patch.object(ToolAgentLoop, "_handle_generating_state", generate):
            await agent._handle_generating_state(data, {})
            await agent._handle_generating_state(data, {})

        self.assertEqual(data.extra_fields["turn_think_token_counts"], [len("First thought"), len("More thought")])


if __name__ == "__main__":
    unittest.main()
