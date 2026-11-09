from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
from grepseek.trainer.verl_integration.hybrid_agent_loop import HybridAgentLoop
from grepseek.trainer.verl_integration.hybrid_dataset import NativeHybridDataset
from grepseek.trainer.verl_integration.reward_function import grepseek_reward
from inference.retrieval_env.schemas import tool_schemas, validate_parameters
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def call_xml(name, parameters):
    return '<tool_call>\n<function=' + name + '>\n' + ''.join(
        '<parameter=' + key + '>\n' + value + '\n</parameter>\n' for key, value in parameters.items()
    ) + '</function>\n</tool_call>'


class NativeHybridTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tokenizer = CharacterTokenizer()
        self.schemas = [OpenAIFunctionToolSchema.model_validate(s) for s in tool_schemas()]
        self.parser = ToolParser.get_tool_parser('qwen3_coder', self.tokenizer)

    async def parse(self, text):
        return (await self.parser.extract_tool_calls(self.tokenizer.encode(text), self.schemas))[1]

    async def test_native_shell_preserves_quotes_backslashes_and_pipe(self):
        command = r'''rg -F "David Dinkins" corpus.jsonl | grep -i "mayor\|born" | head -n 3'''
        calls = await self.parse(call_xml('shell', {'command': command}))
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].arguments), {'command': command})

    async def test_native_dense_and_candidate_scope(self):
        dense = (await self.parse(call_xml('dense_search', {'query': "Marie Curie's spouse?"})))[0]
        self.assertEqual(json.loads(dense.arguments), {'query': "Marie Curie's spouse?"})
        args = {'command': 'cat corpus.jsonl | head -n 3',
                'scope': {'type': 'candidates', 'candidate_set_id': 'handle'}}
        parsed = (await self.parse(call_xml('shell', {'command': args['command'],
                                                     'scope': json.dumps(args['scope'])})))[0]
        self.assertEqual(json.loads(parsed.arguments), args)
        validate_parameters(json.loads(parsed.arguments), tool_schemas()[0]['function']['parameters'])

    async def test_object_parameter_text_is_never_executed(self):
        expression = "__import__('os').system('native-parser-test')"
        with patch('os.system') as execute:
            parsed = (await self.parse(call_xml('shell', {'command': 'cat corpus.jsonl', 'scope': expression})))[0]
        execute.assert_not_called()
        self.assertEqual(json.loads(parsed.arguments)['scope'], expression)
        with self.assertRaises(ValueError):
            validate_parameters(json.loads(parsed.arguments), tool_schemas()[0]['function']['parameters'])

    async def test_schema_injection_is_opt_in(self):
        agent = object.__new__(GrepSeekAgentLoop)
        agent.tool_schemas = tool_schemas()
        agent.apply_chat_template = AsyncMock(return_value=[1, 2])
        data = AgentData([{'role': 'user', 'content': 'Query: Who?'}], None, None, {}, 'test', {})
        data._active_tool_schemas = tool_schemas()
        await agent._handle_pending_state(data, {})
        self.assertEqual(agent.apply_chat_template.await_args.kwargs['tools'], [])
        agent.native_tool_call_format = True
        await agent._handle_pending_state(data, {})
        self.assertEqual(agent.apply_chat_template.await_args.kwargs['tools'], tool_schemas())

    async def test_hybrid_native_prompt_keeps_full_scope_schema(self):
        agent = object.__new__(HybridAgentLoop)
        agent.native_tool_call_format = True
        agent.tools = {'shell': SimpleNamespace(),
                       'dense_search': SimpleNamespace(store=SimpleNamespace(version='test'), ttl_seconds=3600)}
        agent.tool_schemas = []
        agent.prompt_length = 1024
        agent.record_retrieval_rounds = False
        agent.apply_chat_template = AsyncMock(return_value=[1, 2])
        agent._session_context = SimpleNamespace(set=lambda session: None)
        data = AgentData([], None, None, {}, 'test', {})
        fake_session = SimpleNamespace(store=SimpleNamespace(version='test'))
        with patch('grepseek.trainer.verl_integration.hybrid_agent_loop.RetrievalSession', return_value=fake_session):
            await agent._handle_pending_state(data, {})
        tools = agent.apply_chat_template.await_args.kwargs['tools']
        self.assertIn('oneOf', tools[0]['function']['parameters']['properties']['scope'])
        self.assertEqual(tools, tool_schemas())

    def make_agent_and_data(self, calls):
        agent = object.__new__(HybridAgentLoop)
        agent.tokenizer = self.tokenizer
        agent.loop = asyncio.get_running_loop()
        agent.tool_parser = self.parser
        agent.tool_parser_name = 'qwen3_coder'
        agent.tool_limit_behavior = 'truncate_invalid'
        agent.force_final_answer_at_tool_limit = False
        agent.prefill_final_answer_at_tool_limit = False
        agent.response_length = 15360
        agent.max_user_turns = 5
        agent.max_assistant_turns = 6
        agent.interaction_config_file = None
        agent.tools = {s.function.name: SimpleNamespace(tool_schema=s,
                       parameter_schema=next(t['function']['parameters'] for t in tool_schemas()
                                             if t['function']['name'] == s.function.name)) for s in self.schemas}
        data = AgentData([], None, None, {}, 'test', {})
        data.prompt_ids = self.tokenizer.encode('<think>')
        data.response_mask = [0] * len(data.prompt_ids)
        data.response_logprobs = [0.] * len(data.prompt_ids)
        data.user_turns = data.assistant_turns = calls
        data.extra_fields.update(tool_call_attempt_count=calls, tool_counts={'shell': calls})
        return agent, data

    async def generate(self, agent, data, text):
        ids = self.tokenizer.encode(text)
        agent.server_manager = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
            token_ids=ids, log_probs=[-.1] * len(ids), extra_fields={}, num_preempted=None, routed_experts=None)))
        return await agent._handle_generating_state(data, {'temperature': 0})

    async def test_fifth_native_request_remains_dispatchable(self):
        agent, data = self.make_agent_and_data(4)
        state = await self.generate(agent, data, 'Search.</think>' + call_xml('shell', {'command': 'cat corpus.jsonl'}))
        self.assertEqual(state, AgentState.PROCESSING_TOOLS)
        self.assertFalse(data.extra_fields.get('invalid_tool_request', False))

    async def test_sixth_native_request_is_rejected_without_execution(self):
        agent, data = self.make_agent_and_data(5)
        with patch.object(ToolAgentLoop, '_call_tool', new_callable=AsyncMock) as dispatch:
            state = await self.generate(agent, data, 'Search.</think>' + call_xml('shell', {'command': 'cat corpus.jsonl'}))
        dispatch.assert_not_awaited()
        self.assertEqual(state, AgentState.TERMINATED)
        self.assertTrue(data.extra_fields['tool_call_limit_exceeded'])
        self.assertEqual(data.extra_fields['tool_call_attempt_count'], 6)
        self.assertEqual(sum(data.extra_fields['tool_counts'].values()), 5)

    async def test_native_answer_at_budget_zero_keeps_original_score(self):
        agent, data = self.make_agent_and_data(5)
        state = await self.generate(agent, data, 'Done.</think><answer>Paris</answer>')
        self.assertEqual(state, AgentState.TERMINATED)
        score = grepseek_reward('test', self.tokenizer.decode(data.prompt_ids), {'golden_answers': ['Paris']},
                               {**data.extra_fields, 'is_validation': True}, reward_metric='f1', enable_length_decay=False)
        self.assertEqual(score['reward/f1'], 1.)

    def test_native_dataset_keeps_task_and_answer_contract_without_json_examples(self):
        root = Path(__file__).resolve().parents[2]
        original = (root / 'rl/prompts/system_prompt_hybrid.txt').read_text()
        native = NativeHybridDataset._build_system_prompt(None, {
            'system_prompt_path': root / 'rl/prompts/system_prompt_hybrid_native.txt'})
        self.assertTrue(native.startswith(original.split('For every step,')[0].strip()))
        self.assertIn('<answer>...</answer>', native)
        self.assertNotIn('"arguments"', native)


if __name__ == '__main__':
    unittest.main()
