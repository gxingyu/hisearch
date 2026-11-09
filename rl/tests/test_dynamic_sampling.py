from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.dynamic_sampling import filter_reward_groups
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage


def scored_batch(uids, scores):
    size = len(scores)
    return DataProto.from_dict(
        tensors={
            "dummy_tensor": torch.zeros(size, 1, dtype=torch.uint8),
            "input_ids": torch.zeros(size, 3, dtype=torch.long),
            "position_ids": torch.arange(3).repeat(size, 1),
            "prompts": torch.zeros(size, 1, dtype=torch.long),
            "responses": torch.arange(size * 2).reshape(size, 2),
            "attention_mask": torch.ones(size, 3, dtype=torch.long),
            "response_mask": torch.tensor([[0, 1]]).repeat(size, 1),
            "rm_scores": torch.stack((torch.zeros(size), torch.tensor(scores, dtype=torch.float32)), dim=1),
        },
        non_tensors={
            "multi_modal_inputs": np.asarray([{} for _ in range(size)], dtype=object),
            "uid": np.asarray(uids, dtype=object),
            "score": np.asarray(scores),
            "reward/f1": np.maximum(scores, 0),
            "reward/em": np.zeros(size),
            "request_id": np.asarray([f"{uid}:{index % 5}" for index, uid in enumerate(uids)], dtype=object),
        },
        meta_info={"reward_extra_keys": ["score", "reward/f1", "reward/em"]},
    )


def prompt_batch(size=2):
    return {"dummy_tensor": torch.zeros(size, 1, dtype=torch.uint8)}


class FakeTrainLoader:
    sampler = None

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class DynamicSamplingTests(unittest.TestCase):
    def setUp(self):
        self.trainer = object.__new__(RayPPOTrainer)
        self.trainer.config = OmegaConf.create(
            {
                "algorithm": {"filter_groups": {"enable": True, "max_num_gen_batches": 10}},
                "actor_rollout_ref": {"rollout": {"n": 5, "temperature": 1.0}},
                "data": {"train_batch_size": 3},
            }
        )
        self.trainer.global_steps = 7
        self.trainer._training_epoch = 0
        self.trainer.checkpoint_manager = MagicMock()
        self.trainer.async_rollout_manager = MagicMock()

    def test_filter_uses_final_reward_and_preserves_partial_f1_groups(self):
        rewards = [1] * 5 + [0] * 5 + [4 / 9] * 5 + [-1] * 5 + [0, 0.2, 0.4, 0.6, 0.8] + [-1, 0, 0, 0, 0]
        uids = [f"group-{index // 5}" for index in range(len(rewards))]
        batch = scored_batch(uids, rewards)
        filtered, sampled, active = filter_reward_groups(batch, 5)
        self.assertEqual((sampled, active), (6, 2))
        self.assertEqual(Counter(filtered.non_tensor_batch["uid"]), {"group-4": 5, "group-5": 5})
        self.assertTrue(np.all(filtered.non_tensor_batch["reward/em"] == 0))
        np.testing.assert_allclose(filtered.batch["rm_scores"].sum(-1), filtered.non_tensor_batch["score"])

    def test_filter_keeps_groups_intact_when_rows_are_reordered(self):
        batch = scored_batch(["flat"] * 5 + ["active"] * 5, [0] * 5 + [0, 0, 0, 0, 1])
        order = np.asarray([9, 0, 7, 2, 5, 4, 6, 1, 8, 3])
        filtered, sampled, active = filter_reward_groups(batch[order], 5)
        self.assertEqual((sampled, active), (2, 1))
        self.assertEqual(set(filtered.non_tensor_batch["uid"]), {"active"})
        self.assertEqual(set(filtered.non_tensor_batch["request_id"]), {f"active:{index}" for index in range(5)})

    def test_constant_fractional_rewards_produce_no_retained_group(self):
        batch = scored_batch(["flat"] * 5, [4 / 9] * 5)
        filtered, sampled, active = filter_reward_groups(batch, 5)
        self.assertEqual((len(filtered), sampled, active), (0, 1, 0))

    def test_incomplete_groups_and_nonfinite_rewards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 5 rollouts"):
            filter_reward_groups(scored_batch(["group"] * 4, [0, 0, 0, 1]), 5)
        with self.assertRaisesRegex(ValueError, "finite rewards"):
            filter_reward_groups(scored_batch(["group"] * 5, [0, 0, 0, 0, float("nan")]), 5)

    def test_refill_keeps_full_groups_with_one_policy_and_aligned_rewards(self):
        first = scored_batch(["flat"] * 5 + ["active"] * 5, [0] * 5 + [0, 0, 0, 0, 1])
        subsequent_scores = iter([[0.25] * 5 + [0, 0.2, 0.4, 0.6, 0.8], [0, 0, 0, 0, 1] * 2])
        rollout_state = SimpleNamespace(awake=False)

        def restore_weights(global_steps):
            self.assertEqual(global_steps, 7)
            rollout_state.awake = True

        def sleep_replicas():
            self.assertTrue(rollout_state.awake)
            rollout_state.awake = False

        def generate(gen_batch):
            self.assertTrue(rollout_state.awake)
            self.assertEqual(gen_batch.meta_info["global_steps"], 7)
            self.assertEqual(Counter(gen_batch.non_tensor_batch["uid"]).most_common()[0][1], 5)
            output = scored_batch(gen_batch.non_tensor_batch["uid"], next(subsequent_scores))
            output.meta_info["timing"] = {"tool_time": 2.0}
            return output

        self.trainer.checkpoint_manager.update_weights.side_effect = restore_weights
        self.trainer.checkpoint_manager.sleep_replicas.side_effect = sleep_replicas
        self.trainer.async_rollout_manager.generate_sequences.side_effect = generate
        metrics = {}
        timing = {"tool_time": 1.0}
        result = self.trainer._collect_dynamic_sampling_batch(
            first, iter([prompt_batch(), prompt_batch()]), metrics, timing, False
        )
        self.assertEqual(len(result), 15)
        self.assertEqual(set(Counter(result.non_tensor_batch["uid"]).values()), {5})
        self.assertEqual(self.trainer.global_steps, 7)
        self.assertEqual(self.trainer.checkpoint_manager.update_weights.call_args_list, [call(7), call(7)])
        self.assertEqual(self.trainer.checkpoint_manager.sleep_replicas.call_count, 2)
        self.assertFalse(rollout_state.awake)
        self.assertEqual(self.trainer.async_rollout_manager.generate_sequences.call_count, 2)
        self.assertEqual(timing["tool_time"], 5.0)
        self.assertEqual(metrics["dynamic_sampling/generated_batches"], 3)
        self.assertEqual(metrics["dynamic_sampling/sampled_groups"], 6)
        self.assertEqual(metrics["dynamic_sampling/active_groups"], 4)
        self.assertEqual(metrics["dynamic_sampling/kept_groups"], 3)
        self.assertEqual(metrics["dynamic_sampling/filtered_groups"], 2)
        self.assertEqual(metrics["dynamic_sampling/discarded_active_groups"], 1)
        self.assertEqual(metrics["dynamic_sampling/generated_tokens"], 30)
        np.testing.assert_allclose(result.batch["rm_scores"].sum(-1), result.non_tensor_batch["score"])
        result.batch["token_level_rewards"] = result.batch["rm_scores"]
        result = compute_advantage(result, adv_estimator="grpo")
        self.assertTrue(torch.all(result.batch["advantages"][:, 0] == 0))
        for uid in set(result.non_tensor_batch["uid"]):
            indices = np.flatnonzero(result.non_tensor_batch["uid"] == uid)
            advantages = result.batch["advantages"][indices, 1]
            self.assertTrue(torch.any(advantages > 0))
            self.assertTrue(torch.any(advantages < 0))

    def test_full_initial_batch_needs_no_refill(self):
        self.trainer.config.data.train_batch_size = 1
        first = scored_batch(["active"] * 5, [0, 0, 0, 0, 1])
        result = self.trainer._collect_dynamic_sampling_batch(first, iter([]), {}, {}, False)
        self.assertEqual(len(result), 5)
        self.trainer.async_rollout_manager.generate_sequences.assert_not_called()
        self.trainer.checkpoint_manager.update_weights.assert_not_called()

    def test_stop_auxiliary_observes_flat_groups_only_from_first_raw_batch(self):
        first = scored_batch(['flat'] * 5 + ['active'] * 5, [1.] * 5 + [0, 0, 0, 0, 1])
        self.trainer.config.data.train_batch_size = 2
        self.trainer.config.trainer = {'default_local_dir': '/unused/unit-test'}
        self.trainer._stop_controller = MagicMock()
        self.trainer._log_stop_sampling_rewards = MagicMock()
        def prepare(batch, *args, **kwargs):
            self.assertIs(batch, first)
            self.assertEqual(list(batch.non_tensor_batch['uid']), ['flat'] * 5 + ['active'] * 5)
            return [{'qid': 'verified-flat-prefix'}], {'stop_control/auxiliary_examples': 1}
        self.trainer._stop_controller.prepare.side_effect = prepare
        def generate(batch):
            result = scored_batch(batch.non_tensor_batch['uid'], [0, 0, 0, 0, 1])
            result.meta_info['timing'] = {}
            return result
        self.trainer.async_rollout_manager.generate_sequences.side_effect = generate
        metrics = {}
        result = self.trainer._collect_dynamic_sampling_batch(first, iter([prompt_batch(1)]), metrics, {}, False)
        self.assertEqual(len(result), 10)
        self.assertNotIn('flat', result.non_tensor_batch['uid'])
        self.assertEqual(self.trainer._stop_control_payload, [{'qid': 'verified-flat-prefix'}])
        self.trainer._stop_controller.prepare.assert_called_once()
        self.assertEqual(self.trainer._log_stop_sampling_rewards.call_count, 2)
        self.assertEqual(metrics['stop_control/auxiliary_examples'], 1)

    def test_tool_usage_includes_filtered_rollouts(self):
        self.trainer.config.data.train_batch_size = 1
        first = scored_batch(["flat"] * 5 + ["active"] * 5, [0] * 5 + [0, 0, 0, 0, 1])
        first.non_tensor_batch["tool_counts"] = np.asarray(
            [{"dense_search": 2}] * 5 + [{"shell": 1}] * 5, dtype=object
        )
        metrics = {}
        kept = self.trainer._collect_dynamic_sampling_batch(first, iter([]), metrics, {}, False)
        self.assertEqual(len(kept), 5)
        self.assertEqual(metrics["train_all/tool_usage/trajectories"], 10)
        self.assertEqual(metrics["train_all/tool_usage/calls/dense_search"], 10)
        self.assertEqual(metrics["train_all/tool_usage/calls/shell"], 5)
        self.assertTrue(all(record == {"shell": 1} for record in kept.non_tensor_batch["tool_counts"]))

    def test_limit_prevents_infinite_sampling_without_partial_update(self):
        self.trainer.config.algorithm.filter_groups.max_num_gen_batches = 1
        first = scored_batch(["flat"] * 5, [0] * 5)
        with self.assertRaisesRegex(RuntimeError, "0/3 active groups after 1"):
            self.trainer._collect_dynamic_sampling_batch(first, iter([prompt_batch()]), {}, {}, False)
        self.trainer.async_rollout_manager.generate_sequences.assert_not_called()

    def test_epoch_exhaustion_drops_incomplete_batch(self):
        first = scored_batch(["active"] * 5, [0, 0, 0, 0, 1])
        with self.assertLogs("verl.trainer.ppo.ray_trainer", level="WARNING"):
            result = self.trainer._collect_dynamic_sampling_batch(first, iter([]), {}, {}, False)
        self.assertIsNone(result)

    def test_rewards_are_computed_before_filtering_when_not_in_rollout(self):
        self.trainer.config.data.train_batch_size = 1
        first = scored_batch(["active"] * 5, [0, 0, 0, 0, 1])
        reward_output = first.pop(batch_keys=["rm_scores"])
        self.trainer._compute_reward_colocate = MagicMock(return_value=reward_output)
        result = self.trainer._collect_dynamic_sampling_batch(first, iter([]), {}, {}, False)
        self.assertEqual(len(result), 5)
        self.trainer._compute_reward_colocate.assert_called_once()

    def test_enable_switch_defaults_to_false(self):
        self.assertTrue(self.trainer._dynamic_sampling_enabled)
        self.trainer.config.algorithm.filter_groups.enable = False
        self.assertFalse(self.trainer._dynamic_sampling_enabled)
        self.trainer.config.algorithm.filter_groups = None
        self.assertFalse(self.trainer._dynamic_sampling_enabled)

    def test_resume_restores_sampling_epoch_and_cursor_when_switch_changes(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as checkpoint_root:
                checkpoint = Path(checkpoint_root) / "global_step_2"
                checkpoint.mkdir()
                torch.save({"cursor": 1, "dynamic_sampling_epoch": 2}, checkpoint / "data.pt")
                self.trainer.config.algorithm.filter_groups.enable = enabled
                self.trainer.config.trainer = {
                    "resume_mode": "auto",
                    "default_hdfs_dir": None,
                    "default_local_dir": checkpoint_root,
                    "del_local_ckpt_after_load": False,
                }
                self.trainer.actor_rollout_wg = MagicMock()
                self.trainer.use_critic = False
                self.trainer.train_dataloader = FakeTrainLoader([prompt_batch(), prompt_batch()])
                self.trainer.train_dataloader.load_state_dict = MagicMock()
                with patch("verl.trainer.ppo.ray_trainer.find_latest_ckpt_path", return_value=str(checkpoint)):
                    self.trainer._load_checkpoint()
                self.assertEqual(self.trainer.global_steps, 2)
                self.assertEqual(self.trainer._training_epoch, 2)
                self.trainer.train_dataloader.load_state_dict.assert_called_once_with({"cursor": 1})


class DynamicSamplingFitTests(unittest.TestCase):
    def test_fit_switch_and_refill_before_logprobs_and_actor_update(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as output_dir:
                repo_root = Path(__file__).resolve().parents[2]
                environment = {
                    "GREPSEEK_ROOT": str(repo_root),
                    "GREPSEEK_TRAIN_FILES": "unused-train.jsonl",
                    "GREPSEEK_VAL_FILES": "unused-val.jsonl",
                    "GREPSEEK_OUTPUT_DIR": output_dir,
                }
                with patch.dict("os.environ", environment), initialize_config_dir(
                    config_dir=str(repo_root / "rl/grepseek/trainer/config"), version_base=None
                ):
                    config = compose(config_name="grpo_trainer")
                    OmegaConf.resolve(config)
                config.algorithm.filter_groups.enable = enabled
                config.data.train_batch_size = 2
                config.trainer.total_training_steps = 1
                config.trainer.test_freq = 0
                config.trainer.save_freq = -1
                trainer = object.__new__(RayPPOTrainer)
                trainer.config = config
                trainer.train_dataloader = FakeTrainLoader([prompt_batch(), prompt_batch()])
                trainer.train_dataset = SimpleNamespace()
                trainer.total_training_steps = 1
                trainer.use_rm = False
                trainer.use_reference_policy = False
                trainer.use_critic = False
                trainer.resource_pool_manager = MagicMock()
                trainer.resource_pool_manager.get_n_gpus.return_value = 1
                trainer.actor_rollout_wg = SimpleNamespace()
                trainer.checkpoint_manager = MagicMock()
                trainer.async_rollout_manager = MagicMock()
                for method in ("_load_checkpoint", "_start_profiling", "_stop_profiling", "_log_rollout_data"):
                    setattr(trainer, method, MagicMock())
                trainer._balance_batch = lambda batch, metrics: batch.reorder(torch.arange(len(batch) - 1, -1, -1))
                events = []

                def generate(gen_batch):
                    events.append("generate")
                    output = scored_batch(gen_batch.non_tensor_batch["uid"], [0] * 5 + [0, 0, 0, 0, 1])
                    output.meta_info["timing"] = {}
                    return output

                def old_log_prob(batch):
                    events.append("logprobs")
                    return DataProto.from_dict(
                        tensors={"old_log_probs": torch.zeros(len(batch), 2), "entropys": torch.zeros(len(batch), 2)}
                    ), 0.0

                def update_actor(batch):
                    events.append("update")
                    self.assertEqual(len(batch), 10)
                    if enabled:
                        self.assertTrue(torch.all(batch.batch["advantages"][:, 1] != 0))
                    else:
                        self.assertEqual(int((batch.batch["advantages"][:, 1] == 0).sum()), 5)
                    return DataProto(meta_info={"metrics": {"actor/grad_norm": 1.0}})

                trainer.async_rollout_manager.generate_sequences.side_effect = generate
                trainer._compute_old_log_prob = MagicMock(side_effect=old_log_prob)
                trainer._update_actor = MagicMock(side_effect=update_actor)
                with patch("verl.utils.tracking.Tracking"), patch("verl.trainer.ppo.ray_trainer.tqdm"), patch(
                    "verl.trainer.ppo.ray_trainer.compute_data_metrics", return_value={}
                ), patch("verl.trainer.ppo.ray_trainer.compute_variance_proxy_metrics", return_value={}):
                    trainer.fit()
                self.assertEqual(events, ["generate"] * (2 if enabled else 1) + ["logprobs", "update"])
                saved_batch, saved_rewards = trainer._log_rollout_data.call_args.args[:2]
                np.testing.assert_allclose(saved_batch.batch["rm_scores"].sum(-1), saved_rewards["score"])
                self.assertEqual(trainer.global_steps, 2)


if __name__ == "__main__":
    unittest.main()
