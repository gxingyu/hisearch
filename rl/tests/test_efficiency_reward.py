from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
from grepseek.trainer.verl_integration.invalid_tool_request import InvalidToolRequest
from grepseek.trainer.verl_integration.reward_function import grepseek_efficiency_reward, grepseek_reward
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop


class EfficiencyRewardTests(unittest.TestCase):
    answer = "<think>The retrieved evidence answers this.</think><answer>Abraham Lincoln</answer>"
    gold = {"golden_answers": ["Abraham Lincoln"]}

    def reward(self, *, calls=1, generated=100, text=None, **extra):
        return grepseek_efficiency_reward(
            "test", self.answer if text is None else text, self.gold,
            {"tool_call_attempt_count": calls, "generated_token_count": generated,
             "response_token_count": max(generated, 5000), **extra},
        )

    def test_correct_answers_prefer_fewer_calls(self):
        for calls, expected in [(0, 1), (1, 1), (2, .98), (3, .96), (4, .94), (5, .92)]:
            with self.subTest(calls=calls):
                result = self.reward(calls=calls)
                self.assertAlmostEqual(result["score"], expected)
                self.assertEqual(result["reward/f1"], 1)

    def test_partial_credit_is_discounted_and_wrong_calls_stay_flat(self):
        partial = self.answer.replace("Abraham Lincoln", "Lincoln")
        self.assertAlmostEqual(self.reward(calls=5, text=partial)["score"], (2 / 3) * .92)
        wrong = self.answer.replace("Abraham Lincoln", "Washington")
        for calls in range(6):
            self.assertEqual(self.reward(calls=calls, text=wrong)["score"], 0)

    def test_generated_buffer_boundaries_and_no_zero_clamp(self):
        for generated, penalty in [(0, 0), (512, 0), (1280, .1), (2048, .2), (10000, .2)]:
            result = self.reward(calls=5, generated=generated)
            self.assertAlmostEqual(result["reward/generation_penalty"], penalty)
            self.assertAlmostEqual(result["score"], .92 - penalty)
        wrong = self.answer.replace("Abraham Lincoln", "Washington")
        self.assertAlmostEqual(self.reward(text=wrong, generated=2048)["score"], -.2)

    def test_observation_volume_does_not_change_score(self):
        first = self.reward(calls=3, response_token_count=1000)
        second = self.reward(calls=3, response_token_count=15000)
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(second["metrics/non_generated_response_token_count"], 14900)

    def test_invalid_and_missing_answer_use_minus_one_without_extra_penalties(self):
        for text in ["", "<think>Still searching", "<think>Unsure</think><answer></answer>", self.answer + "junk"]:
            result = self.reward(text=text, generated=10000, calls=5)
            self.assertEqual(result["score"], -1)
            self.assertEqual(result["reward/call_penalty"], 0)
            self.assertEqual(result["reward/generation_penalty"], 0)
        self.assertEqual(self.reward(invalid_tool_request=True)["score"], -1)

    def test_validation_matches_every_original_reward_metric(self):
        for text in [self.answer, self.answer.replace("Abraham Lincoln", "Lincoln"), "<think>truncated",
                     "<think>Unsure</think><answer></answer>"]:
            extra = {"tool_call_attempt_count": 5, "generated_token_count": 2048,
                     "response_token_count": 10000, "is_validation": True}
            expected = grepseek_reward("test", text, self.gold, extra, reward_metric="f1", enable_length_decay=False)
            actual = grepseek_efficiency_reward("test", text, self.gold, extra)
            for key, value in expected.items():
                self.assertEqual(actual[key], value, key)
            self.assertEqual(actual["reward/call_penalty"], 0)
            self.assertEqual(actual["reward/generation_penalty"], 0)

    def test_missing_or_invalid_exact_metadata_fails_loudly(self):
        for extra in [{}, {"tool_call_attempt_count": 1},
                      {"tool_call_attempt_count": True, "generated_token_count": 1},
                      {"tool_call_attempt_count": -1, "generated_token_count": 1},
                      {"tool_call_attempt_count": 1, "generated_token_count": 1.5},
                      {"tool_call_attempt_count": 1, "generated_token_count": 2, "response_token_count": 1}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                grepseek_efficiency_reward("test", self.answer, self.gold, extra)

    def test_legacy_length_settings_do_not_stack(self):
        result = grepseek_efficiency_reward(
            "test", self.answer, self.gold,
            {"tool_call_attempt_count": 5, "generated_token_count": 1280, "response_token_count": 15000},
            enable_length_decay=True, a=.9, enable_penalty_warmup=True,
        )
        self.assertAlmostEqual(result["score"], .82)

    def test_zero_call_cost_preserves_buffer_for_correct_partial_and_wrong_answers(self):
        for prediction, f1 in [("Abraham Lincoln", 1.0), ("Lincoln", 2 / 3), ("Washington", 0.0)]:
            for generated, penalty in [(0, 0), (512, 0), (513, .2 / 1536),
                                       (1280, .1), (2048, .2), (10000, .2)]:
                for calls in range(6):
                    with self.subTest(prediction=prediction, generated=generated, calls=calls):
                        result = grepseek_efficiency_reward(
                            "test", self.answer.replace("Abraham Lincoln", prediction), self.gold,
                            {"tool_call_attempt_count": calls, "generated_token_count": generated,
                             "response_token_count": 15000}, call_cost_rate=0.0,
                        )
                        self.assertAlmostEqual(result["score"], f1 - penalty)
                        self.assertEqual(result["reward/call_penalty"], 0)
                        self.assertEqual(result["reward/call_discount"], 1)
                        self.assertAlmostEqual(result["reward/generation_penalty"], penalty)

    def test_zero_call_cost_removes_cost_only_difference_in_same_correct_group(self):
        scores = [grepseek_efficiency_reward(
            "test", self.answer, self.gold,
            {"tool_call_attempt_count": calls, "generated_token_count": 100}, call_cost_rate=0.0,
        )["score"] for calls in [2, 1, 1, 1, 1]]
        self.assertEqual(scores, [1.0] * 5)

    def test_zero_call_cost_preserves_validation_and_failure_handling(self):
        for text in [self.answer, self.answer.replace("Abraham Lincoln", "Lincoln"),
                     "<think>truncated", "<think>Unsure</think><answer></answer>"]:
            for is_validation in [False, True]:
                extra = {"tool_call_attempt_count": 5, "generated_token_count": 10000,
                         "response_token_count": 15000, "is_validation": is_validation}
                result = grepseek_efficiency_reward("test", text, self.gold, extra, call_cost_rate=0.0)
                self.assertEqual(result["reward/call_penalty"], 0)
                if is_validation:
                    original = grepseek_reward("test", text, self.gold, extra,
                                               reward_metric="f1", enable_length_decay=False)
                    for key, value in original.items():
                        self.assertEqual(result[key], value, key)
                    self.assertEqual(result["reward/generation_penalty"], 0)
                elif not result["reward/format_pass"] or result["reward/failed_rollout"]:
                    self.assertEqual(result["score"], -1)
                    self.assertEqual(result["reward/generation_penalty"], 0)


class EfficiencyAgentMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_mask_counts_generation_without_observations_or_injected_prefix(self):
        agent = object.__new__(GrepSeekAgentLoop)
        agent.response_length = 100
        agent.max_user_turns = 5
        output = SimpleNamespace(response_ids=list(range(8)), response_mask=[1, 1, 0, 0, 0, 1, 0, 0],
                                 extra_fields={"tool_call_attempt_count": 2})
        with patch.object(ToolAgentLoop, "run", AsyncMock(return_value=output)):
            actual = await agent.run({})
        self.assertIs(actual, output)
        self.assertEqual(actual.extra_fields["generated_token_count"], 3)
        self.assertEqual(actual.extra_fields["response_token_count"], 8)
        self.assertEqual(actual.extra_fields["tool_call_attempt_count"], 2)

    async def test_empty_and_failed_tool_results_both_count_without_changing_result(self):
        agent = object.__new__(GrepSeekAgentLoop)
        agent.force_final_answer_at_tool_limit = False
        data = AgentData([], None, None, {}, "test", {})
        for result in [(SimpleNamespace(text=""), 0, {"exit_code": 0}),
                       (SimpleNamespace(text="error"), 0, {"exit_code": -2})]:
            with patch.object(ToolAgentLoop, "_call_tool", AsyncMock(return_value=result)):
                self.assertEqual(await agent._call_tool(None, {}, data), result)
        self.assertEqual(data.extra_fields["tool_call_attempt_count"], 2)

    async def test_terminal_invalid_request_counts_once(self):
        agent = object.__new__(GrepSeekAgentLoop)
        agent.apply_chat_template = AsyncMock()
        agent.response_length = 100
        data = AgentData([], None, None, {}, "test", {})
        with patch("grepseek.trainer.verl_integration.agent_loop.append_terminal_invalid_tool_response", AsyncMock()):
            state = await agent._terminate_invalid_tool_request(data, InvalidToolRequest("malformed_tool_call", "bad"))
        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(data.extra_fields["tool_call_attempt_count"], 1)


if __name__ == "__main__":
    unittest.main()
