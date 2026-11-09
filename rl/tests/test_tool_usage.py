import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from verl.trainer.ppo.tool_usage import extend_tool_usage, summarize_tool_usage, tool_usage_columns


class ToolUsageTests(unittest.TestCase):
    def test_counts_scopes_and_fractions(self):
        columns = tool_usage_columns(
            {
                "tool_counts": np.asarray([{"dense_search": 1, "shell": 2}, {"shell": 1}], dtype=object),
                "tool_trace": np.asarray(
                    [
                        [
                            {"tool_name": "dense_search"},
                            {"tool_name": "shell", "scope": "corpus"},
                            {"tool_name": "shell", "scope": "candidates"},
                        ],
                        [{"tool_name": "shell", "scope": "corpus"}],
                    ],
                    dtype=object,
                ),
            }
        )
        metrics = summarize_tool_usage(columns, "val/tool_usage")
        self.assertEqual(metrics["val/tool_usage/calls/total"], 4)
        self.assertEqual(metrics["val/tool_usage/calls/shell_candidates"], 1)
        self.assertAlmostEqual(metrics["val/tool_usage/call_fraction/dense_search"], 0.25)
        self.assertAlmostEqual(metrics["val/tool_usage/trajectory_fraction/mixed"], 0.5)

    def test_missing_counts_are_unavailable(self):
        self.assertEqual(tool_usage_columns({}), {})
        self.assertEqual(summarize_tool_usage({}, "train/tool_usage"), {})

    def test_zero_calls_and_unknown_scope(self):
        columns = tool_usage_columns({"tool_counts": [{}, {"shell": 1}]})
        metrics = summarize_tool_usage(columns, "train/tool_usage")
        self.assertEqual(metrics["train/tool_usage/calls/shell_unknown_scope"], 1)
        self.assertEqual(metrics["train/tool_usage/trajectory_fraction/no_tool"], 0.5)
        empty = summarize_tool_usage(tool_usage_columns({"tool_counts": [{}, {}]}), "val/tool_usage")
        self.assertEqual(empty["val/tool_usage/call_fraction/dense_search"], 0.0)
        self.assertEqual(empty["val/tool_usage/trajectory_fraction/no_tool"], 1.0)

    def test_validation_benchmarks_and_merged_batches_use_summed_counts(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        columns = tool_usage_columns({"tool_counts": [{"dense_search": 4}, {"shell": 1}]})
        extend_tool_usage(columns, tool_usage_columns({"tool_counts": [{}]}))
        columns["reward"] = [1.0, 0.0, 0.0]
        trainer = object.__new__(RayPPOTrainer)
        metrics = trainer._val_metrics_update(np.asarray(["qa", "qa", "other"]), ["a", "b", "c"], columns, [])
        self.assertEqual(metrics["val/tool_usage/trajectories"], 3)
        self.assertEqual(metrics["val/tool_usage/call_fraction/dense_search"], 0.8)
        self.assertEqual(metrics["val/tool_usage_by_benchmark/qa/calls/total"], 5)
        self.assertEqual(metrics["val/tool_usage_by_benchmark/other/trajectory_fraction/no_tool"], 1.0)

    def test_validation_excludes_distributed_padding(self):
        from omegaconf import OmegaConf

        from rl.tests.test_dynamic_sampling import scored_batch
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = scored_batch(["a", "b", "c"], [1, 0, 0])
        batch.non_tensor_batch["data_source"] = np.asarray(["qa"] * 3, dtype=object)
        batch.non_tensor_batch["tool_counts"] = np.asarray([{"dense_search": 4}, {"shell": 1}, {}], dtype=object)
        trainer = object.__new__(RayPPOTrainer)
        trainer.global_steps = 120
        trainer.use_rm = False
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {"val_kwargs": {"n": 1, "do_sample": False}, "agent": {"num_workers": 2}}
                },
                "trainer": {"log_val_generations": 0},
            }
        )
        trainer.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0, decode=lambda *args, **kwargs: "text")
        trainer.val_dataloader = [{**dict(batch.batch), **batch.non_tensor_batch}]
        trainer.async_rollout_manager = MagicMock()

        def generate(padded):
            self.assertEqual(len(padded), 4)
            return batch.select_idxs(np.asarray([0, 1, 2, 0]))

        trainer.async_rollout_manager.generate_sequences.side_effect = generate
        metrics = trainer._validate()
        self.assertEqual(metrics["val/tool_usage/trajectories"], 3)
        self.assertEqual(metrics["val/tool_usage/calls/total"], 5)
        self.assertEqual(metrics["val/tool_usage/call_fraction/dense_search"], 0.8)


if __name__ == "__main__":
    unittest.main()
