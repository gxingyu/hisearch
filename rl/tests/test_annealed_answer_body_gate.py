"""Regression tests for C's actual gate, credit budgets and GSPO gradients."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from verl import DataProto
from verl.trainer.ppo.answer_body_gate import AnswerBodyGate, annealed_gate_weight, bounded_soft_gate
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config.actor import ActorConfig

spec = importlib.util.spec_from_file_location("hard_gate_fixtures", Path(__file__).with_name("test_answer_body_gate.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class AnnealedGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = AnswerBodyGate(fixtures.CharacterTokenizer(), mode="annealed_soft")
        self.batch = fixtures.example_batch()

    def score(self, probabilities):
        def fn(jobs):
            n = len(jobs)
            return DataProto.from_dict(tensors={"answer_body_logp": torch.tensor(
                [math.log(x) for x in probabilities] * (n // 5), dtype=torch.float64)})
        return fn

    def test_restored_global_step_schedule(self):
        self.assertEqual(annealed_gate_weight(0), 0)
        self.assertEqual(annealed_gate_weight(40), 0)
        self.assertAlmostEqual(annealed_gate_weight(41), .001541333133436018)
        self.assertAlmostEqual(annealed_gate_weight(60), .5)
        self.assertEqual(annealed_gate_weight(80), 1)
        self.assertEqual(annealed_gate_weight(200), 1)
        self.assertTrue(all(annealed_gate_weight(k) <= annealed_gate_weight(k+1) for k in range(200)))

    def test_warmup_has_no_forward_no_tensor_changes(self):
        before = {k:v.clone() for k,v in self.batch.batch.items()}
        def forbidden(_): self.fail("Before step41 no gate scoring forward is needed")
        m = self.gate.apply(self.batch, forbidden, step=40)
        for k,v in before.items():torch.testing.assert_close(self.batch.batch[k],v,rtol=0,atol=0)
        self.assertEqual(m['answer_body_gate/annealing_weight'],0)
        self.assertEqual(m['answer_body_gate/positive_credit_removed_fraction'],0)

    def test_protect_negative_zero_first_round_answer_and_non_advantage_tensors(self):
        self.batch.batch['advantages'][0,16] = 0
        before = {k:v.clone() for k,v in self.batch.batch.items()}
        with tempfile.TemporaryDirectory() as out:
            m = self.gate.apply(self.batch, self.score([1e-7,1e-10,.999,.9991,.9992]),step=80,output_dir=out)
            self.assertTrue((Path(out)/'answer_body_gate/80.summary.json').is_file())
        original=before['advantages'];after=self.batch.batch['advantages']
        changed=original!=after
        self.assertTrue(bool(changed.any()))
        self.assertTrue(bool((after[original>0]>0).all()))
        self.assertFalse(bool(changed[1].any()))
        self.assertFalse(bool(changed[:,:15].any()))
        self.assertFalse(bool(changed[:,23:].any()))
        for k,v in before.items():
            if k!='advantages':torch.testing.assert_close(self.batch.batch[k],v,rtol=0,atol=0)
        self.assertLessEqual(m['answer_body_gate/positive_credit_removed_fraction'],.01+1e-7)
        self.assertEqual(m['answer_body_gate/fully_cleared_generated_tokens'],0)

    def test_actual_gspo_gradient_is_reduced_only_on_selected_tokens(self):
        cfg=ActorConfig(strategy='fsdp',rollout_n=5,ppo_micro_batch_size_per_gpu=1)
        old=self.batch.batch['old_log_probs']
        before=self.batch.batch['advantages'].clone()
        def grad(a):
            x=old.clone().requires_grad_(True)
            loss,_=compute_policy_loss_gspo(old,x,a,self.batch.batch['response_mask'],config=cfg)
            return torch.autograd.grad(loss,x)[0]
        g0=grad(before)
        self.gate.apply(self.batch,self.score([.999]*5),step=60)
        after=self.batch.batch['advantages'];g1=grad(after);changed=before!=after
        self.assertTrue(bool(changed.any()))
        self.assertTrue(bool((g1[changed].abs()<g0[changed].abs()).all()))
        self.assertTrue(bool((g1[changed]!=0).all()))
        torch.testing.assert_close(g1[~changed],g0[~changed],rtol=0,atol=0)

    def test_large_drop_and_necessary_gain_are_preserved(self):
        before=self.batch.batch['advantages'].clone()
        self.gate.apply(self.batch,self.score([.986937,.999970,.00333,.99,.2]),step=80)
        torch.testing.assert_close(self.batch.batch['advantages'],before,rtol=0,atol=0)

    def test_budget_is_global_not_per_dp_chunk_and_annealing_survives_saturation(self):
        a=torch.tensor([[1.,1.,1.,1.],[2.,2.,2.,2.],[-1.,-1.,-1.,-1.],[3.,3.,3.,3.]])
        mask=torch.ones_like(a);selected=torch.tensor([[1,1,0,0],[0,0,0,0],[1,1,0,0],[0,0,0,0]],dtype=torch.bool)
        full,m,*_=bounded_soft_gate(a,mask,selected,weight=1)
        half,hm,*_=bounded_soft_gate(a,mask,selected,weight=.5)
        self.assertLess(m['budget_kappa'],1)
        torch.testing.assert_close(a-half,.5*(a-full),atol=1e-7,rtol=0)
        self.assertAlmostEqual(m['positive_credit_removed_fraction'],.01,places=7)
        self.assertAlmostEqual(hm['positive_credit_removed_fraction'],.005,places=7)
        chunks=[bounded_soft_gate(a[j:j+2],mask[j:j+2],selected[j:j+2],weight=1)[0] for j in [0,2]]
        self.assertFalse(torch.allclose(full,torch.cat(chunks)))

    def test_empty_failed_trajectory_and_no_eligible_round(self):
        a=torch.tensor([[0.,0.],[1.,1.],[-1.,-1.]])
        mask=torch.tensor([[0,0],[1,1],[1,1]])
        out,m,*_=bounded_soft_gate(a,mask,torch.zeros_like(mask,dtype=torch.bool),weight=1)
        torch.testing.assert_close(out,a,rtol=0,atol=0)
        self.assertEqual(m['softened_generated_tokens'],0)

    def test_variable_lengths_random_caps_and_negative_protection(self):
        gen=torch.Generator().manual_seed(20261029)
        for _ in range(100):
            a=torch.randn((20,64),generator=gen)
            mask=torch.arange(64)[None,:]<torch.randint(0,65,(20,1),generator=gen)
            a=a*mask;chosen=(torch.rand(a.shape,generator=gen)<.5)&mask
            out,m,*_=bounded_soft_gate(a,mask,chosen,weight=.5)
            self.assertTrue(torch.equal(out[a<=0],a[a<=0]))
            self.assertTrue(bool((out[a>0]>0).all()))
            self.assertLessEqual(m['positive_credit_removed_fraction'],.005+1e-7)
            self.assertLessEqual(m['max_sequence_positive_credit_removed_fraction'],.05+1e-7)
            self.assertLessEqual(m['max_effective_token_removal'],.125+1e-12)

    def test_invalid_settings_and_observation_selection_fail(self):
        for kwargs in [{'mode':'unknown'},{'anneal_start_step':80,'anneal_end_step':40},
                       {'max_token_removal':1},{'max_batch_credit_removal':float('nan')},
                       {'anneal_start_step':True}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                AnswerBodyGate(fixtures.CharacterTokenizer(),**kwargs)
        with self.assertRaises(ValueError):
            bounded_soft_gate(torch.ones(1,2),torch.tensor([[1,0]]),torch.tensor([[False,True]]),weight=1)


class PilotEndpointTests(unittest.TestCase):
    def test_real_trainer_stops_at80_with_optimizer_schedule200(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from rl.tests.test_dynamic_sampling import FakeTrainLoader, prompt_batch, scored_batch
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as out, patch.dict('os.environ',{
            'GREPSEEK_ROOT':str(root),'GREPSEEK_TRAIN_FILES':'unused-train.jsonl',
            'GREPSEEK_VAL_FILES':'unused-val.jsonl','GREPSEEK_OUTPUT_DIR':out}):
            with initialize_config_dir(config_dir=str(root/'rl/grepseek/trainer/config'),version_base=None):
                config=compose(config_name='grpo_trainer');OmegaConf.resolve(config)
            config.algorithm.filter_groups.enable=False
            config.data.train_batch_size=2
            config.trainer.total_training_steps=200
            config.actor_rollout_ref.actor.optim.total_training_steps=200
            config.trainer.stop_after_step=80
            config.trainer.save_freq=5;config.trainer.test_freq=10
            trainer=object.__new__(RayPPOTrainer);trainer.config=config
            trainer.train_dataloader=FakeTrainLoader([prompt_batch(),prompt_batch()])
            trainer.train_dataset=SimpleNamespace();trainer.total_training_steps=200
            trainer.use_rm=trainer.use_reference_policy=trainer.use_critic=False
            trainer.resource_pool_manager=MagicMock();trainer.resource_pool_manager.get_n_gpus.return_value=1
            trainer.actor_rollout_wg=SimpleNamespace();trainer.checkpoint_manager=MagicMock()
            trainer.async_rollout_manager=MagicMock()
            for method in ['_start_profiling','_stop_profiling','_log_rollout_data',
                           '_save_checkpoint','_maybe_update_best_checkpoint','_prune_checkpoints']:
                setattr(trainer,method,MagicMock())
            def restore():
                trainer.global_steps=79
                trainer._training_epoch=0  # Saved dynamic_sampling_epoch, not step/loader-length.
            trainer._load_checkpoint=MagicMock(side_effect=restore)
            trainer._balance_batch=lambda batch,metrics:None
            def generate(batch):
                output=scored_batch(batch.non_tensor_batch['uid'],[0]*5+[0,0,0,0,1])
                output.meta_info['timing']={};return output
            trainer.async_rollout_manager.generate_sequences.side_effect=generate
            trainer._compute_old_log_prob=MagicMock(side_effect=lambda b:(DataProto.from_dict(tensors={
                'old_log_probs':torch.zeros(len(b),2),'entropys':torch.zeros(len(b),2)}),0.))
            trainer._update_actor=MagicMock(return_value=DataProto(meta_info={'metrics':{'actor/grad_norm':1.}}))
            trainer._validate=MagicMock(return_value={'val/benchmarks/average_f1':.6})
            with patch('verl.utils.tracking.Tracking'),patch('verl.trainer.ppo.ray_trainer.tqdm'),patch(
                'verl.trainer.ppo.ray_trainer.compute_data_metrics',return_value={}),patch(
                'verl.trainer.ppo.ray_trainer.compute_variance_proxy_metrics',return_value={}):
                trainer.fit()
            self.assertEqual(trainer._update_actor.call_count,1)
            self.assertEqual(trainer._save_checkpoint.call_count,1)
            self.assertEqual(trainer._validate.call_count,1)
            self.assertEqual(trainer.total_training_steps,200)
            self.assertEqual(config.actor_rollout_ref.actor.optim.total_training_steps,200)


if __name__=='__main__':unittest.main()
