from __future__ import annotations

import asyncio
import json
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from grepseek.trainer.verl_integration.reward_function import grepseek_stop_judge_reward
from grepseek.trainer.verl_integration.stop_judge import JudgeConfig, JudgeError, StopJudge, parse_prefixes


def trajectory(count=3, answer='James Madison'):
    parts = []
    for index in range(count):
        parts.append(f'<think>The answer is James Madison step {index + 1}.</think>'
                     '<tool_call>{"name":"dense_search","arguments":{"query":"president"}}</tool_call>'
                     '\nuser\n<tool_response>' + json.dumps({'documents': [f'visible {index + 1}']})
                     + '</tool_response>\nassistant\n')
    return ''.join(parts) + f'<think>Done.</think><answer>{answer}</answer>'


def response(status='ready', quote='James Madison', usage=None):
    verdict = {'status': status, 'answer': 'James Madison' if status == 'ready' else '',
               'reasoning_quote': quote if status == 'ready' else '', 'reason_code': 'test'}
    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(verdict)}}],
                                   'usage': usage or {'prompt_tokens': 100, 'completion_tokens': 20}})


class PrefixTests(unittest.TestCase):
    def test_prefix_excludes_current_result_and_future(self):
        prefixes = parse_prefixes(trajectory())
        self.assertEqual(len(prefixes), 3)
        self.assertEqual(prefixes[0]['history'], '')
        self.assertNotIn('visible 1', json.dumps(prefixes[0]))
        self.assertIn('visible 1', prefixes[1]['history'])
        self.assertNotIn('visible 2', json.dumps(prefixes[1]))
        self.assertNotIn('<answer>', json.dumps(prefixes))

    def test_unpaired_and_invalid_calls_fail(self):
        for text in [trajectory() + '<tool_call>', trajectory().replace('"dense_search"', '"unknown"'),
                     trajectory().replace('"documents"', 'broken', 1)]:
            with self.assertRaises(JudgeError):
                parse_prefixes(text)

    def test_first_thought_may_have_no_opening_tag(self):
        self.assertIn('James Madison', parse_prefixes(trajectory().removeprefix('<think>'))[0]['reasoning'])


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_template_and_reasoning_audit(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertNotIn('enable_thinking', payload)
            self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
            reply = response()
            body = reply.json()
            body['choices'][0]['message']['reasoning'] = 'unexpected reasoning'
            return httpx.Response(200, json=body)

        judge = StopJudge(JudgeConfig(thinking_parameter='chat_template_kwargs'), api_key='test',
                          transport=httpx.MockTransport(handler))
        try:
            await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(judge.events[0]['separate_reasoning_chars'], len('unexpected reasoning'))
        finally:
            await judge.close()

    async def test_first_ready_stops_and_counts_current_call(self):
        calls = []

        async def handler(request):
            calls.append(json.loads(request.content))
            return response('not_ready' if len(calls) == 1 else 'ready')

        judge = StopJudge(JudgeConfig(), api_key='test', transport=httpx.MockTransport(handler))
        try:
            result = await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(result['first_ready_tool_index'], 2)
            self.assertEqual(result['extra_tool_calls'], 2)
            self.assertEqual(result['api_calls'], 2)
            self.assertEqual(result['input_tokens'], 200)
            self.assertNotIn('<answer>', json.dumps(calls))
            self.assertEqual(calls[0]['enable_thinking'], False)
            self.assertEqual(calls[0]['response_format'], {'type': 'json_object'})
        finally:
            await judge.close()

    async def test_no_ready_uncertain_is_not_error(self):
        judge = StopJudge(JudgeConfig(), api_key='test', transport=httpx.MockTransport(lambda request: response('uncertain')))
        try:
            result = await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(result['first_ready_tool_index'], -1)
            self.assertEqual(result['extra_tool_calls'], 0)
            self.assertEqual(result['api_calls'], 3)
            self.assertEqual(result['uncertain_prefixes'], 3)
        finally:
            await judge.close()

    async def test_inline_thinking_only_parses_final_json(self):
        judge = StopJudge(JudgeConfig(), api_key='test', transport=httpx.MockTransport(lambda request: response()))
        try:
            content = 'Analysis with {bad json}</think>' + json.dumps({
                'status': 'ready', 'answer': 'James Madison', 'reasoning_quote': 'James Madison', 'reason_code': 'complete',
            })
            self.assertEqual(judge.validate(content, 'James Madison')['status'], 'ready')
            with self.assertRaises(JudgeError):
                judge.validate('Analysis {"status":"ready"}', 'James Madison')
        finally:
            await judge.close()

    async def test_non_verbatim_ready_quote_is_uncertain_without_penalty(self):
        judge = StopJudge(JudgeConfig(retries=0), api_key='test',
                          transport=httpx.MockTransport(lambda request: response(quote='not in reasoning')))
        try:
            result = await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(result['first_ready_tool_index'], -1)
            self.assertEqual(result['extra_tool_calls'], 0)
            self.assertEqual(result['quote_mismatches'], 3)
            self.assertEqual(result['uncertain_prefixes'], 3)
            self.assertTrue(all(event['verdict']['status'] == 'uncertain' for event in judge.events))
            self.assertTrue(all(event['error'] == '' for event in judge.events))
        finally:
            await judge.close()

    async def test_uncertain_prefix_does_not_prevent_later_valid_ready(self):
        replies = iter([response(quote='paraphrased answer'), response()])
        judge = StopJudge(JudgeConfig(), api_key='test',
                          transport=httpx.MockTransport(lambda request: next(replies)))
        try:
            result = await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(result['first_ready_tool_index'], 2)
            self.assertEqual(result['extra_tool_calls'], 2)
            self.assertEqual(result['uncertain_prefixes'], 1)
            self.assertEqual(result['quote_mismatches'], 1)
        finally:
            await judge.close()

    async def test_ready_missing_fields_is_uncertain(self):
        judge = StopJudge(JudgeConfig(), api_key='test', transport=httpx.MockTransport(lambda request: response(quote='')))
        try:
            for field in ('answer', 'reasoning_quote'):
                verdict = {'status': 'ready', 'answer': 'James Madison',
                           'reasoning_quote': 'James Madison', 'reason_code': 'test'}
                verdict[field] = ''
                result = judge.validate(json.dumps(verdict), 'James Madison')
                self.assertEqual(result['status'], 'uncertain')
            result = await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(result['extra_tool_calls'], 0)
            self.assertEqual(result['uncertain_prefixes'], 3)
        finally:
            await judge.close()

    async def test_auth_failure_still_fails_closed(self):
        judge = StopJudge(JudgeConfig(retries=0), api_key='test',
                          transport=httpx.MockTransport(lambda request: httpx.Response(401, text='sensitive body')))
        try:
            with self.assertRaises(JudgeError) as raised:
                await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertNotIn('sensitive', str(raised.exception))
        finally:
            await judge.close()

    async def test_retry_records_usage_without_double_counting_reasoning(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429)
            return response(usage={'prompt_tokens': 100, 'completion_tokens': 30,
                                   'completion_tokens_details': {'reasoning_tokens': 10},
                                   'prompt_tokens_details': {'cached_tokens': 50}})

        judge = StopJudge(JudgeConfig(retries=1), api_key='test', transport=httpx.MockTransport(handler))
        try:
            with patch('grepseek.trainer.verl_integration.stop_judge.asyncio.sleep', return_value=None):
                result = await judge.score('Who?', ['James Madison'], parse_prefixes(trajectory()))
            self.assertEqual(result['api_calls'], 2)
            self.assertEqual(result['retries'], 1)
            self.assertEqual(result['output_tokens'], 30)
            self.assertEqual(result['reasoning_tokens'], 10)
        finally:
            await judge.close()

    async def test_concurrency_bound_and_singleflight(self):
        active = 0
        maximum = 0

        async def handler(request):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return response()

        judge = StopJudge(JudgeConfig(concurrency=2), api_key='test', transport=httpx.MockTransport(handler))
        prefix = parse_prefixes(trajectory())[0]
        try:
            await asyncio.gather(*(judge.judge(f'Who {index}?', ['James Madison'], prefix) for index in range(8)))
            self.assertEqual(maximum, 2)
            before = len(judge.events)
            await asyncio.gather(*(judge.judge('identical', ['James Madison'], prefix) for _ in range(8)))
            self.assertEqual(len(judge.events) - before, 1)
        finally:
            await judge.close()

    async def test_persistent_cache_and_audit_omit_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            config = JudgeConfig(cache_dir=directory, audit_path=str(Path(directory) / 'audit.jsonl'))
            judge = StopJudge(config, api_key='secret-credential', transport=httpx.MockTransport(lambda request: response()))
            try:
                prefix = parse_prefixes(trajectory())[0]
                await judge.judge('Who?', ['James Madison'], prefix)
                _, metrics = await judge.judge('Who?', ['James Madison'], prefix)
                self.assertEqual(metrics['cache_hits'], 1)
                await judge.judge('Who?', ['Different gold'], prefix)
                self.assertEqual(len(judge.events), 2)
                self.assertNotIn('secret-credential', ''.join(path.read_text() for path in Path(directory).iterdir()))
            finally:
                await judge.close()


class RewardTests(unittest.IsolatedAsyncioTestCase):
    async def reward(self, output=None, **extra):
        return await grepseek_stop_judge_reward(
            'test', trajectory() if output is None else output, {'golden_answers': ['James Madison']},
            {'question': 'Who?', **extra}, stop_judge_enabled=True,
            reward_metric='f1', enable_length_decay=False,
        )

    async def test_penalty_not_clamped_and_final_wrong_not_exempt(self):
        fake = type('FakeJudge', (), {})()

        async def score(*args):
            return {'first_ready_tool_index': 1, 'extra_tool_calls': 3, 'api_calls': 1}

        fake.score = score
        with patch('grepseek.trainer.verl_integration.stop_judge.get_judge', return_value=fake):
            correct = await self.reward()
            wrong = await self.reward(trajectory(answer='wrong'))
        self.assertAlmostEqual(correct['score'], 0.4)
        self.assertAlmostEqual(correct['reward/judge'], -0.6)
        self.assertAlmostEqual(wrong['score'], -0.6)
        self.assertEqual(wrong['reward/f1'], 0)

    async def test_uncertain_judge_has_no_stop_penalty(self):
        fake = type('FakeJudge', (), {})()

        async def score(*args):
            return {'first_ready_tool_index': -1, 'extra_tool_calls': 0,
                    'quote_mismatches': 1, 'api_calls': 1}

        fake.score = score
        with patch('grepseek.trainer.verl_integration.stop_judge.get_judge', return_value=fake):
            result = await self.reward()
        self.assertEqual(result['judge/triggered'], 0)
        self.assertEqual(result['judge/quote_mismatches'], 1)
        self.assertEqual(result['reward/stop_penalty'], 0.0)
        self.assertEqual(result['reward/judge'], 0.0)
        self.assertEqual(result['score'], 1.0)

    async def test_validation_disabled_invalid_and_natural_answer_skip_api(self):
        with patch('grepseek.trainer.verl_integration.stop_judge.get_judge') as client:
            validation = await self.reward(is_validation=True)
            invalid = await self.reward('invalid')
            natural = await self.reward('<think>Done.</think><answer>James Madison</answer>')
            client.assert_not_called()
        self.assertEqual(validation['score'], 1)
        self.assertEqual(invalid['score'], -1)
        self.assertEqual(natural['score'], 1)
        self.assertEqual(set(validation), set(invalid))
        self.assertEqual(set(validation), set(natural))

    async def test_judge_error_propagates(self):
        with patch('grepseek.trainer.verl_integration.stop_judge.get_judge', side_effect=JudgeError('unavailable')):
            with self.assertRaises(JudgeError):
                await self.reward()


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_qwen_config_migration(self):
        from hydra import compose, initialize_config_dir
        from verl.experimental.reward_loop import migrate_legacy_reward_impl

        root = Path(__file__).resolve().parents[2]
        with patch.dict(os.environ, {'GREPSEEK_ROOT': str(root)}):
            with initialize_config_dir(config_dir=str(root / 'rl/grepseek/trainer/config'), version_base=None):
                config = compose(config_name='grpo_trainer', overrides=['+stop_judge@_global_=qwen'])
            config = migrate_legacy_reward_impl(config)
            settings = config.reward.custom_reward_function.reward_kwargs
            self.assertEqual(config.reward.custom_reward_function.name, 'grepseek_stop_judge_reward')
            self.assertEqual(config.reward.num_workers * settings.stop_judge_concurrency, 128)
            self.assertEqual(settings.stop_judge_thinking_parameter, 'chat_template_kwargs')
            self.assertEqual(settings.stop_judge_model, 'qwen38-27b-stop-judge')

    async def test_hydra_async_reward_manager_and_failure_propagation(self):
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from verl import DataProto
        from verl.experimental.reward_loop import migrate_legacy_reward_impl
        from verl.experimental.reward_loop.reward_loop import RewardLoopWorker, reward_secret_runtime_env
        from verl.trainer.ppo.reward import get_custom_reward_fn, load_reward_manager

        root = Path(__file__).resolve().parents[2]
        with patch.dict(os.environ, {'GREPSEEK_ROOT': str(root), 'GREPSEEK_STOP_JUDGE_API_KEY': 'test-secret'}):
            with initialize_config_dir(config_dir=str(root / 'rl/grepseek/trainer/config'), version_base=None):
                config = compose(config_name='grpo_trainer', overrides=['+stop_judge@_global_=enabled'])
            config = migrate_legacy_reward_impl(config)
            config.reward.custom_reward_function.path = str(root / 'rl/grepseek/trainer/verl_integration/reward_function.py')
            config.reward.custom_reward_function.reward_kwargs.stop_judge_cache_dir = ''
            config.reward.custom_reward_function.reward_kwargs.stop_judge_audit_path = ''
            OmegaConf.resolve(config.reward.custom_reward_function)
            self.assertTrue(inspect.iscoroutinefunction(get_custom_reward_fn(config)))
            self.assertEqual(reward_secret_runtime_env(config)['env_vars']['GREPSEEK_STOP_JUDGE_API_KEY'], 'test-secret')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Missing reward credential'):
                reward_secret_runtime_env(config)

        class Tokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return trajectory()

        manager = load_reward_manager(config, Tokenizer())
        self.assertTrue(manager.is_async_reward_score)
        data = DataProto.from_dict(
            tensors={'responses': torch.ones(1, 3, dtype=torch.long),
                     'attention_mask': torch.ones(1, 4, dtype=torch.long)},
            non_tensors={'data_source': np.array(['test']),
                         'reward_model': np.array([{'ground_truth': {'golden_answers': ['James Madison']}}], dtype=object),
                         'extra_info': np.array([{'question': 'Who?'}], dtype=object)},
        )

        class FakeJudge:
            async def score(self, *args):
                return {'first_ready_tool_index': 2, 'extra_tool_calls': 2}

        with patch('grepseek.trainer.verl_integration.stop_judge.get_judge', return_value=FakeJudge()):
            result = await manager.run_single(data)
        self.assertAlmostEqual(result['reward_score'], 0.6)
        self.assertEqual(result['reward_extra_info']['judge/first_ready_tool_index'], 2)
        worker = object.__new__(RewardLoopWorker)
        worker.config = config
        worker.reward_manager = manager
        with patch('grepseek.trainer.verl_integration.stop_judge.get_judge', side_effect=JudgeError('unavailable')):
            with self.assertRaises(JudgeError):
                await worker.compute_score_batch(data)


if __name__ == '__main__':
    unittest.main()
