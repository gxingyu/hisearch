"""Behavioral tests of selection, prefix isolation and distributed loss weight."""
import copy
from contextlib import nullcontext
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from verl import DataProto
from verl.trainer.ppo.answer_body_gate import HEADER, object_array
from verl.trainer.ppo.stop_control import VerifiedStopControl, auxiliary_slots
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig, PolicyLossConfig
from test_answer_body_gate import CharacterTokenizer, example_batch


def raw_batch():
    original = example_batch()
    rows = []
    for group, rewards in enumerate(([1.] * 5, [0.] * 5, [0., 1., 0., 1., 0.])):
        for roll, reward in enumerate(rewards):
            row = copy.deepcopy(original[roll % 2:roll % 2 + 1])
            row.non_tensor_batch['uid'] = object_array([f'u{group}'])
            row.non_tensor_batch['extra_info'] = object_array([{'qid': f'q{group}', 'golden_answers': ['Paris']}])
            row.batch['rm_scores'] = torch.tensor([[reward]])
            rows.append(row)
    return DataProto.concat(rows)


class StopControlTests(unittest.TestCase):
    def run_prepare(self, batch=None, probs=None, include_flat=True, bad_second=False):
        batch = raw_batch() if batch is None else batch
        probabilities = probs or [1e-5, 1e-6, .999, .999, .999]
        self.requests = []
        def score(jobs):
            return DataProto.from_dict(tensors={'answer_body_logp': torch.tensor(
                [math.log(probabilities[i % 5]) for i in range(len(jobs))], dtype=torch.float64)})
        def generate(requests):
            self.requests = copy.deepcopy(requests)
            return [{'token_ids': list(map(ord, ('Wrong' if bad_second and i % 2 else 'Paris') + '</answer>'))}
                    for i in range(len(requests))]
        ctl = VerifiedStopControl(CharacterTokenizer(), {'include_flat': include_flat})
        return ctl.prepare(batch, score, generate, step=41)

    def test_flat_correct_and_flat_wrong_can_both_supply_verified_prefix(self):
        batch = raw_batch()
        before = {k: v.clone() for k, v in batch.batch.items()}
        jobs, metrics = self.run_prepare(batch)
        self.assertEqual(len(jobs), 3)
        self.assertEqual({j['group_kind'] for j in jobs}, {'flat_score1', 'flat_score0', 'active'})
        self.assertEqual(metrics['stop_control/flat_verified_qids'], 2)
        self.assertAlmostEqual(metrics['stop_control/nominal_mixture_weight'], .05 * 3 / 1280)
        for key, value in before.items():
            torch.testing.assert_close(batch.batch[key], value, rtol=0, atol=0)
        # Stop at h3 BEFORE the fourth call, including neither that call nor its observation.
        for job in jobs:
            self.assertEqual(job['prefix_index'], 3)
            self.assertEqual(job['prefix_ids'], [42, 43] + list(range(15)))
            self.assertEqual(job['target_ids'], list(map(ord, HEADER)))
        for request in self.requests:
            self.assertEqual(request['input_ids'], [42, 43] + list(range(15)) + list(map(ord, HEADER)))
            self.assertNotIn('Paris', CharacterTokenizer().decode(request['input_ids']))
        self.assertEqual([r['temperature'] for r in self.requests], [0., 1.] * 3)

    def test_necessary_low_probability_bridge_and_large_drop_get_no_auxiliary(self):
        for probabilities in ([1e-5] * 5, [.999, .1, .01, .001, .0001]):
            jobs, _ = self.run_prepare(probs=probabilities)
            self.assertEqual(jobs, [])
            self.assertEqual(self.requests, [])

    def test_both_answer_probes_required(self):
        jobs, m = self.run_prepare(bad_second=True)
        self.assertEqual(jobs, [])
        self.assertEqual(m['stop_control/candidate_qids'], 3)
        self.assertEqual(m['stop_control/verified_qids'], 0)

    def test_c_ablation_does_not_replace_excluded_flat_groups(self):
        jobs, m = self.run_prepare(include_flat=False)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['group_kind'], 'active')
        self.assertEqual(m['stop_control/verified_qids'], 3)

    def test_selector_is_deterministic_and_does_not_consume_main_rng(self):
        torch_before = torch.get_rng_state()
        np_before = np.random.get_state()
        first, _ = self.run_prepare()
        second, _ = self.run_prepare()
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        for a, b in zip(np_before, np.random.get_state()):
            np.testing.assert_equal(a, b)

    def test_reject_broken_generated_observation_span(self):
        batch = raw_batch()
        batch.batch['response_mask'][0, 3] = 1
        with self.assertRaisesRegex(RuntimeError, 'observation'):
            self.run_prepare(batch)

    def test_uneven_fsdp_jobs_and_weight_relative_to_eight_main_updates(self):
        for count in (1, 7, 8, 9, 31, 32):
            dispatched = []
            main_weight, aux_weight = 0., 0.
            for mb in range(8):
                lengths = []
                for rank in range(8):
                    slots = auxiliary_slots(count, 8, 8, rank, mb)
                    lengths.append(len(slots))
                    dispatched += [s for s in slots if s is not None]
                    aux_weight += sum(s is not None for s in slots) * .05 / 20 / 8
                    main_weight += 20 / 20 / 8
                self.assertEqual(len(set(lengths)), 1)
            self.assertEqual(sorted(dispatched), list(range(count)))
            self.assertAlmostEqual(aux_weight / main_weight, .05 * count / 1280)

    def test_actual_header_loss_matches_shifted_causal_ce_and_gradient(self):
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 8)
                self.head = torch.nn.Linear(8, 64)
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                # Causal dependency on prefix, without adding prefix prediction targets.
                hidden = self.embedding(input_ids).cumsum(1)
                return SimpleNamespace(logits=self.head(hidden)[:, -logits_to_keep:])
        actor = object.__new__(DataParallelPPOActor)
        actor.actor_module = TinyLM()
        actor.device_name, actor.param_dtype = 'cpu', torch.bfloat16
        job = {'prefix_ids': [2, 3, 4], 'target_ids': [5, 6, 7]}
        with patch('verl.workers.actor.dp_actor.get_device_id', return_value='cpu'):
            actual = actor.stop_control_header_loss(job)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = actor.actor_module(torch.tensor([[2, 3, 4, 5, 6]]), None, False, 3).logits[0]
        expected = F.cross_entropy(logits.float(), torch.tensor([5, 6, 7]))
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(sum(float(p.grad.norm()) for p in actor.actor_module.parameters()), 0.)

    def test_actual_actor_loop_calibration_has_zero_steps_and_formal_has_eight(self):
        for calibration in (True, False):
            actor = object.__new__(DataParallelPPOActor)
            actor.config = FSDPActorConfig(strategy='fsdp', rollout_n=5, ppo_mini_batch_size=2,
                ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False, ppo_epochs=1,
                use_kl_loss=False, entropy_coeff=0., grad_clip=1., policy_loss=PolicyLossConfig(loss_mode='gspo'))
            actor.actor_module = torch.nn.Linear(1, 1, bias=False)
            actor.actor_module.weight.data.fill_(.1)
            actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=.001)
            actor.scaler = None
            actor.use_prefix_grouper = actor.use_ulysses_sp = actor.use_fused_kernels = False
            actor._forward_micro_batch = lambda inputs, **kw: {
                'log_probs': actor.actor_module.weight.expand_as(inputs['advantages'])}
            actor.stop_control_header_loss = lambda job: actor.actor_module.weight.square().mean()
            data = DataProto.from_dict(tensors={
                'responses': torch.ones(16, 3, dtype=torch.long),
                'response_mask': torch.ones(16, 3), 'input_ids': torch.ones(16, 4, dtype=torch.long),
                'attention_mask': torch.ones(16, 4), 'position_ids': torch.zeros(16, 4, dtype=torch.long),
                'old_log_probs': torch.full((16, 3), .1), 'advantages': torch.ones(16, 3),
            }, meta_info={'temperature': 1., 'stop_control_jobs': [{}] * 3,
                'stop_control_coefficient': .05, 'stop_control_calibration_only': calibration})
            before = actor.actor_module.weight.detach().clone()
            with patch('verl.workers.actor.dp_actor.get_device_id', return_value='cpu'), \
                 patch('torch.random.fork_rng', return_value=nullcontext()):
                metrics = actor.update_policy(data)
            self.assertEqual(metrics['actor/stop_control_examples_applied'], 3)
            self.assertEqual(metrics['actor/stop_control_optimizer_updates'], 0 if calibration else 8)
            if calibration:
                torch.testing.assert_close(actor.actor_module.weight, before, rtol=0, atol=0)
                self.assertEqual(metrics['actor/stop_control_gradient_ratio_samples'], 3)
                self.assertGreater(metrics['actor/stop_control_gradient_ratio_median'], 0.)
                self.assertEqual(actor.actor_optimizer.state, {})
            else:
                self.assertFalse(torch.equal(actor.actor_module.weight, before))
                self.assertEqual(float(next(iter(actor.actor_optimizer.state.values()))['step']), 8.)


if __name__ == '__main__':
    unittest.main()
