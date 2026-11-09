from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
from grepseek.trainer.verl_integration.hybrid_agent_loop import HybridAgentLoop
from grepseek.trainer.verl_integration.reward_function import grepseek_reward
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


class ToolLimitInvalidTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.agent = object.__new__(HybridAgentLoop)
        self.agent.tokenizer = CharacterTokenizer()
        self.agent.loop = asyncio.get_running_loop()
        self.agent.tool_limit_behavior = "truncate_invalid"
        self.agent.force_final_answer_at_tool_limit = False
        self.agent.prefill_final_answer_at_tool_limit = False
        self.agent.prompt_length = 1024
        self.agent.response_length = 15360
        self.agent.max_user_turns = 5
        self.agent.max_assistant_turns = 6
        self.agent.max_parallel_calls = 1
        self.agent.interaction_config_file = None
        self.agent.tools = {name: SimpleNamespace(tool_schema={"name": name}) for name in ("dense_search", "shell")}

        async def parse(ids, tools):
            text = self.agent.tokenizer.decode(ids)
            try:
                raw = text.split("<tool_call>")[1].split("</tool_call>")[0]
                call = json.loads(raw)
                calls = [FunctionCall(name=call["name"], arguments=json.dumps(call["arguments"]))]
            except (IndexError, ValueError):
                calls = []
            return "", calls

        self.agent.tool_parser = SimpleNamespace(extract_tool_calls=AsyncMock(side_effect=parse))

    def data(self, calls):
        data = AgentData([], None, None, {}, "test", {})
        data.prompt_ids = self.agent.tokenizer.encode("<think>")
        data.response_mask = [0] * len(data.prompt_ids)
        data.response_logprobs = [0.0] * len(data.prompt_ids)
        data.user_turns = data.assistant_turns = calls
        data.extra_fields.update(tool_call_attempt_count=calls, tool_counts={"dense_search": calls}, tool_trace=[])
        data.retrieval_session = SimpleNamespace(store=SimpleNamespace(version="test"))
        return data

    async def generate(self, data, text):
        ids = self.agent.tokenizer.encode(text)
        self.agent.server_manager = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            token_ids=ids, log_probs=[-0.1] * len(ids), extra_fields={}, num_preempted=None, routed_experts=None
        )))
        params = {"temperature": 1.0}
        state = await self.agent._handle_generating_state(data, params)
        self.assertEqual(self.agent.server_manager.generate.await_args.kwargs["sampling_params"], params)
        return state

    def reward(self, data, validation=False):
        extra = {**data.extra_fields, "is_validation": validation, "tool_call_limit": 5}
        return grepseek_reward("test", self.agent.tokenizer.decode(data.prompt_ids),
                               {"golden_answers": ["Paris"]}, extra, reward_metric="f1", enable_length_decay=False)

    async def test_natural_answer_before_or_at_limit_keeps_f1(self):
        for calls in (0, 4, 5):
            with self.subTest(calls=calls):
                data = self.data(calls)
                state = await self.generate(data, "Evidence.</think><answer>Paris</answer>")
                self.assertEqual(state, AgentState.TERMINATED)
                self.assertEqual(self.reward(data)["score"], 1)
                self.assertFalse(data.extra_fields.get("invalid_tool_request", False))
                self.assertTrue(data.extra_fields["natural_answer_emitted"])
                self.assertEqual(data.user_turns, calls)

    async def test_sixth_request_survives_parent_turn_termination_and_gets_minus_one(self):
        for name, args in (("dense_search", {"query": "Paris"}), ("shell", {"command": "cat corpus.jsonl"})):
            with self.subTest(tool=name):
                data = self.data(5)
                text = "Search.</think><tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
                with patch.object(ToolAgentLoop, "_call_tool", new_callable=AsyncMock) as dispatch:
                    state = await self.generate(data, text)
                dispatch.assert_not_awaited()
                self.assertEqual(state, AgentState.TERMINATED)
                self.assertTrue(data.extra_fields["tool_call_limit_exceeded"])
                self.assertEqual(data.extra_fields["tool_call_attempt_count"], 6)
                self.assertEqual(sum(data.extra_fields["tool_counts"].values()), 5)
                self.assertEqual(data.user_turns, 5)
                self.assertEqual(self.agent.tokenizer.decode(data.prompt_ids), "<think>" + text)
                self.assertEqual(data.response_mask, [0] * 7 + [1] * len(text))
                self.assertEqual(len(data.prompt_ids), len(data.response_logprobs))
                self.assertEqual(self.reward(data)["score"], -1)
                self.assertEqual(self.reward(data)["metrics/invalid_tool_request_reason_index"], 7)
                self.assertEqual(self.reward(data, validation=True)["score"], 0)

    async def test_malformed_tool_attempt_and_answer_tool_mixture_are_terminal(self):
        for text in ("Search.</think><tool_call>{broken", "Search.</think></tool_call>",
                     "Done.</think><answer>Paris</answer><tool_call>{broken"):
            with self.subTest(text=text):
                data = self.data(5)
                await self.generate(data, text)
                self.assertTrue(data.extra_fields["tool_call_limit_exceeded"])
                self.assertEqual(self.reward(data)["score"], -1)

    async def test_dispatch_guard_never_executes_sixth_request(self):
        data = self.data(5)
        data.response_ids = self.agent.tokenizer.encode('<tool_call>{"name":"shell"}</tool_call>')
        data.tool_calls = [FunctionCall(name="shell", arguments='{"command":"cat corpus.jsonl"}')]
        with patch.object(ToolAgentLoop, "_handle_processing_tools_state", new_callable=AsyncMock) as process:
            state = await self.agent._handle_processing_tools_state(data)
        process.assert_not_awaited()
        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(data.extra_fields["tool_call_attempt_count"], 6)

    async def test_fifth_call_executes_without_forced_answer_and_counts_failures(self):
        for name, result in (("dense_search", {"status": "ok"}), ("shell", {"exit_code": 2})):
            with self.subTest(tool=name):
                data = self.data(4)
                call = FunctionCall(name=name, arguments=json.dumps({"query": "Paris"} if name == "dense_search"
                                                                  else {"command": "cat corpus.jsonl"}))
                response = ToolResponse(text='{"stdout":""}')
                with patch.object(ToolAgentLoop, "_call_tool", new_callable=AsyncMock,
                                  return_value=(response, 0, result)) as dispatch:
                    response, _, _ = await self.agent._call_tool(call, {}, data)
                dispatch.assert_awaited_once()
                self.assertEqual(data.extra_fields["tool_call_attempt_count"], 5)
                self.assertEqual(sum(data.extra_fields["tool_counts"].values()), 5)
                self.assertEqual(json.loads(response.text)["remaining_tool_calls"], 0)
                self.assertNotIn("final_answer_required", json.loads(response.text))
                self.assertFalse(data.extra_fields.get("force_final_answer_next_turn", False))

    async def test_no_answer_and_response_budget_exhaustion_follow_original_format_rule(self):
        for text in ("", "Still thinking", "Search.</think><tool_call>{broken"):
            with self.subTest(text=text):
                data = self.data(5)
                self.agent.response_length = len(data.response_mask) + len(text)
                await self.generate(data, text)
                self.assertEqual(self.reward(data)["score"], -1)
                self.assertEqual(len(data.prompt_ids), len(data.response_mask))

    def test_opt_in_disables_all_forced_answer_controls(self):
        def initialize(agent, *args, **kwargs):
            agent.tokenizer = CharacterTokenizer()
            agent.tools = {}
        with patch.object(ToolAgentLoop, "__init__", initialize):
            agent = GrepSeekAgentLoop(tool_limit_behavior="truncate_invalid")
            legacy = GrepSeekAgentLoop()
        self.assertFalse(agent.force_final_answer_at_tool_limit)
        self.assertFalse(agent.prefill_final_answer_at_tool_limit)
        self.assertTrue(legacy.force_final_answer_at_tool_limit)
        self.assertTrue(legacy.prefill_final_answer_at_tool_limit)


class AgentRegistrationTests(unittest.TestCase):
    def test_cold_import_keeps_treatment_for_every_trajectory(self):
        # The Hybrid class decorator registers its legacy name at import time.
        # Test in fresh interpreters: an already-imported test class hides this.
        root = Path(__file__).resolve().parents[2]
        script = '''
import json,sys
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
import hydra
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(map(ord,text))
def initialize(agent,*args,**kwargs):
    agent.tokenizer=Tokenizer();agent.tools={}
config=OmegaConf.load(sys.argv[1])[0]
name=sys.argv[2]
config.name=name
_agent_loop_registry[name]=config
modes=[]
with patch.object(ToolAgentLoop,'__init__',initialize):
    for _ in range(2):
        agent=hydra.utils.instantiate(_agent_loop_registry[name])
        modes.append(agent.tool_limit_behavior)
print(json.dumps(modes))
'''
        cfg = root / "rl/grepseek/trainer/config/agent_loops_hybrid_tool_limit_invalid.yaml"
        for name, expected in (
            ("grepseek_hybrid_agent", ["truncate_invalid", "force_answer"]),
            ("grepseek_hybrid_tool_limit_invalid_agent", ["truncate_invalid", "truncate_invalid"]),
        ):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, "-c", script, str(cfg), name],
                                        cwd=root, env=os.environ.copy(), capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                self.assertEqual(json.loads(result.stdout.splitlines()[-1]), expected)


if __name__ == "__main__":
    unittest.main()
