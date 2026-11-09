from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import unittest
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from grepseek.rubric.controller import RubricController, object_array, validate_training_configuration
from grepseek.rubric.core import RubricError, digest, load_snapshot
from grepseek.rubric.client import RubricClient
from rl.tests.test_dynamic_sampling import scored_batch, prompt_batch
from rl.tests.test_rubric_workflow import INITIAL_PATH, JudgeServer, trajectory
from verl.trainer.ppo.dynamic_sampling import filter_reward_groups
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage


def rubric_batch(uids, passes):
    batch=scored_batch(uids,[.6]*len(uids))
    records=[trajectory(str(uid),str(uid),i,passed=passed) for i,(uid,passed) in enumerate(zip(uids,passes))]
    batch.non_tensor_batch.update({
        'extra_info':object_array([{'qid':r['qid'],'question':r['question'],'is_validation':False} for r in records]),
        'rubric_events':object_array([r['events'][1:] for r in records]),
        'retrieval_request_id':object_array([r['trajectory_id'] for r in records]),
        'reward/format_pass':np.ones(len(uids)),
        'reward/failed_rollout':np.zeros(len(uids)),
    })
    batch.meta_info['reward_extra_keys']+=['reward/format_pass','reward/failed_rollout']
    return batch


class RubricIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.env=patch.dict(os.environ,{'GREPSEEK_RUBRIC_API_KEY':'test-only-token','GREPSEEK_OUTPUT_DIR':''})
        self.env.start()
        self.server=JudgeServer()
        self.client=RubricClient({'base_url':'http://rubric.invalid/v1','model':'mock','retries':0},
                                 transport=httpx.MockTransport(self.server))
        self.controller=RubricController({'initial_path':str(INITIAL_PATH),'output_dir':str(self.root/'rubric')},client=self.client)
        self.trainer=object.__new__(RayPPOTrainer)
        self.trainer.config=OmegaConf.create({
            'algorithm':{'filter_groups':{'enable':True,'max_num_gen_batches':3}},
            'actor_rollout_ref':{'rollout':{'n':5,'temperature':1.0},'actor':{'checkpoint':{'async_save':False}}},
            'data':{'train_batch_size':1},
            'trainer':{'default_local_dir':str(self.root/'training'),'default_hdfs_dir':None,
                       'max_actor_ckpt_to_keep':None,'max_critic_ckpt_to_keep':None,
                       'resume_mode':'disable','del_local_ckpt_after_load':False}})
        self.trainer.global_steps=1
        self.trainer._training_epoch=0
        self.trainer._rubric_controller=self.controller
        self.trainer._stop_controller=None
        self.trainer.async_rollout_manager=MagicMock()
        self.trainer.checkpoint_manager=MagicMock()
        self.trainer.actor_rollout_wg=MagicMock()
        self.trainer.use_critic=False
        self.trainer.train_dataloader=MagicMock()
        self.trainer.train_dataloader.__len__.return_value=10
        self.trainer.train_dataloader.state_dict.return_value={'cursor':1}

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_rubric_revives_equal_f1_group_before_actual_dynamic_filter(self):
        first=rubric_batch(['flat']*5+['mixed']*5,[True]*5+[True,False,False,False,False])
        self.assertEqual(filter_reward_groups(first,5)[2],0)
        self.controller.begin_step(1)
        metrics={}
        kept=self.trainer._collect_dynamic_sampling_batch(first,iter([]),metrics,{},False)
        self.assertEqual(set(kept.non_tensor_batch['uid']),{'mixed'})
        self.assertEqual(metrics['dynamic_sampling/sampled_groups'],2)
        self.assertEqual(len(kept),5)
        np.testing.assert_allclose(kept.batch['rm_scores'].sum(-1),kept.non_tensor_batch['score'])
        np.testing.assert_allclose(kept.non_tensor_batch['reward/f1'],[.6]*5)
        kept.batch['token_level_rewards']=kept.batch['rm_scores']
        kept=compute_advantage(kept,adv_estimator='grpo')
        self.assertGreater(float(kept.batch['advantages'][0,1]),0)
        self.assertTrue(torch.all(kept.batch['advantages'][1:,1]<0))
        self.assertTrue(torch.all(kept.batch['advantages'][:,0]==0))
        metrics=self.controller.finish_step(kept)
        self.assertEqual(metrics['rubric/all_sampled/trajectories'],10)
        self.assertEqual(metrics['rubric/retained/trajectories'],5)

    def test_refill_freezes_snapshot_and_observes_all_discarded_groups(self):
        self.trainer.config.data.train_batch_size=2
        first=rubric_batch(['flat']*5+["first-active"]*5,[True]*5+[True,False,False,False,False])
        def generate(batch):
            result=rubric_batch(batch.non_tensor_batch['uid'],[True,False,False,False,False])
            result.meta_info['timing']={}
            return result
        self.trainer.async_rollout_manager.generate_sequences.side_effect=generate
        self.controller.begin_step(1)
        metrics={}
        kept=self.trainer._collect_dynamic_sampling_batch(first,iter([prompt_batch(1)]),metrics,{},False)
        self.assertEqual(len(kept),10)
        self.assertEqual(set(kept.non_tensor_batch['rubric/version']),{0})
        self.assertEqual(metrics['dynamic_sampling/generated_batches'],2)
        self.assertEqual(len(self.controller.workflow.records),15)
        all_versions={r['rubric_version'] for r in self.controller.workflow.records}
        self.assertEqual(all_versions,{0})
        result=self.controller.finish_step(kept)
        self.assertEqual(result['rubric/all_sampled/trajectories'],15)
        self.assertEqual(result['rubric/retained/trajectories'],10)

    def test_actual_trainer_checkpoint_saves_and_restores_matching_rubric_state(self):
        self.controller.begin_step(1)
        batch=self.controller.score_batch(rubric_batch(['q']*5,[True,False,False,False,False]))
        self.controller.finish_step(batch)
        self.trainer._save_checkpoint()
        checkpoint=self.root/'training/global_step_1'
        self.assertTrue((checkpoint/'rubric_state.json').is_file())
        state=json.loads((checkpoint/'rubric_state.json').read_text())
        self.assertEqual(state['completed_step'],1)
        restored=RubricController({'initial_path':str(INITIAL_PATH),'output_dir':str(self.root/'restored')},client=self.client)
        self.trainer._rubric_controller=restored
        self.trainer.config.trainer.resume_mode='resume_path'
        self.trainer.config.trainer.resume_from_path=str(checkpoint)
        self.trainer._load_checkpoint()
        self.assertEqual(restored.workflow.completed_step,1)
        self.assertEqual(restored.begin_step(2)['version'],0)

    def test_missing_checkpoint_state_is_rejected_before_loading_policy(self):
        checkpoint=self.root/'training/global_step_3'
        checkpoint.mkdir(parents=True)
        self.trainer.config.trainer.resume_mode='resume_path'
        self.trainer.config.trainer.resume_from_path=str(checkpoint)
        with self.assertRaisesRegex(RubricError,'rubric state'):
            self.trainer._load_checkpoint()
        self.trainer.actor_rollout_wg.load_checkpoint.assert_not_called()

    def test_invalid_and_failed_answers_keep_base_score_and_validation_is_excluded(self):
        batch=rubric_batch(['q']*5,[True]*5)
        batch.non_tensor_batch['reward/format_pass'][0]=0
        batch.batch['rm_scores'][0,-1]=-1
        batch.non_tensor_batch['reward/f1'][0]=0
        batch.non_tensor_batch['reward/failed_rollout'][1]=1
        batch.batch['rm_scores'][1,-1]=0
        batch.non_tensor_batch['reward/f1'][1]=0
        self.controller.begin_step(1)
        result=self.controller.score_batch(batch)
        self.assertEqual(float(result.batch['rm_scores'][0].sum()),-1)
        self.assertEqual(float(result.batch['rm_scores'][1].sum()),0)
        self.assertEqual(result.non_tensor_batch['reward/rubric'][0],0)
        self.assertEqual(result.non_tensor_batch['reward/rubric'][1],0)
        other=rubric_batch(['validation']*5,[True]*5)
        other.non_tensor_batch['extra_info'][0]['is_validation']=True
        with self.assertRaisesRegex(RubricError,'Validation'):
            self.controller.score_batch(other)

    def test_missing_visible_ledger_and_extra_reward_are_not_silently_accepted(self):
        self.controller.begin_step(1)
        batch=rubric_batch(['q']*5,[True]*5)
        del batch.non_tensor_batch['rubric_events']
        with self.assertRaisesRegex(RubricError,'visible token ledger'):self.controller.score_batch(batch)
        batch=rubric_batch(['q']*5,[True]*5)
        batch.batch['rm_scores'][0,-1]-=.2
        with self.assertRaisesRegex(RubricError,'raw F1'):self.controller.score_batch(batch)

    def test_epoch_end_drops_partial_collection_without_advancing_rubric(self):
        self.controller.begin_step(1)
        batch=rubric_batch(['flat']*5,[True]*5)
        result=self.trainer._collect_dynamic_sampling_batch(batch,iter([]),{}, {},False)
        self.assertIsNone(result)
        self.assertIsNone(self.controller.workflow.step)
        self.assertEqual(self.controller.workflow.completed_step,0)
        self.assertEqual(self.controller.begin_step(1)['version'],0)

    def test_config_defaults_disable_rubrics_and_explicit_recipe_rejects_mixed_objectives(self):
        root=Path(__file__).resolve().parents[2]
        with patch.dict(os.environ,{'GREPSEEK_ROOT':str(root),'GREPSEEK_OUTPUT_DIR':str(self.root/'run'),
                                   'GREPSEEK_TRAIN_FILES':'train.jsonl','GREPSEEK_VAL_FILES':'val.jsonl'}):
            with initialize_config_dir(config_dir=str(root/'rl/grepseek/trainer/config'),version_base=None):
                cfg=compose(config_name='grpo_trainer')
            self.assertFalse(cfg.algorithm.rubric.enabled)
            cfg.algorithm.rubric.enabled=True
            cfg.actor_rollout_ref.actor.policy_loss.loss_mode='gspo'
            validate_training_configuration(cfg)
            cfg.algorithm.route_efficiency.enabled=True
            with self.assertRaisesRegex(RubricError,'route_efficiency'):validate_training_configuration(cfg)

    def test_hybrid_exports_only_final_delivered_token_ledger_when_enabled(self):
        from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
        from grepseek.trainer.verl_integration.hybrid_agent_loop import HybridAgentLoop
        agent=object.__new__(HybridAgentLoop)
        agent.config=OmegaConf.create({'algorithm':{'rubric':{'enabled':True}}})
        agent._session_context=ContextVar('rubric-test-session',default=None)
        agent.tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:''.join(chr(c) for c in ids))
        async def parent_run(*args,**kwargs):
            return SimpleNamespace(response_ids=list(map(ord,'askshownanswer')),response_mask=[1]*3+[0]*5+[1]*6,
                                   extra_fields={'invalid_tool_request':False})
        with patch.object(GrepSeekAgentLoop,'run',parent_run):
            result=asyncio.run(agent.run({},extra_info={'qid':'q'}))
        events=result.extra_fields['rubric_events']
        self.assertEqual([e['text'] for e in events],['ask','shown','answer'])
        self.assertEqual(events[-1]['kind'],'answer')


if __name__=='__main__':unittest.main()
