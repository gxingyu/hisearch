from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class CheckpointRetentionTests(unittest.TestCase):
    def test_keeps_latest_and_best_as_best_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            trainer = object.__new__(RayPPOTrainer)
            trainer.config = OmegaConf.create(
                {
                    "trainer": {
                        "default_local_dir": temp_dir,
                        "keep_latest_and_best_checkpoint": True,
                        "best_checkpoint": {
                            "enable": True,
                            "metric": "val/benchmarks/average_f1",
                            "mode": "max",
                            "name": "best_checkpoint",
                        },
                    }
                }
            )
            trainer._best_checkpoint_score = None

            for step, score in ((40, 0.5), (80, 0.4), (120, 0.6)):
                checkpoint_dir = output_dir / f"global_step_{step}"
                (checkpoint_dir / "actor").mkdir(parents=True)
                (checkpoint_dir / "data.pt").touch()
                (output_dir / "latest_checkpointed_iteration.txt").write_text(str(step))
                trainer.global_steps = step
                trainer._maybe_update_best_checkpoint({"val/benchmarks/average_f1": score})
                trainer._prune_checkpoints()

                expected_steps = {40, step} if step < 120 else {120}
                actual_steps = {
                    int(path.name.removeprefix("global_step_"))
                    for path in output_dir.glob("global_step_*")
                    if path.is_dir()
                }
                self.assertEqual(actual_steps, expected_steps)
                best_metadata = json.loads((output_dir / "best_checkpoint.json").read_text())
                self.assertTrue((output_dir / best_metadata["checkpoint"] / "actor").is_dir())

    def test_invalid_tracker_does_not_delete_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for step in (40, 80):
                (output_dir / f"global_step_{step}").mkdir()
            (output_dir / "latest_checkpointed_iteration.txt").write_text("missing")
            trainer = object.__new__(RayPPOTrainer)
            trainer.config = OmegaConf.create(
                {"trainer": {"default_local_dir": temp_dir, "keep_latest_and_best_checkpoint": True}}
            )

            trainer._prune_checkpoints()

            self.assertTrue((output_dir / "global_step_40").is_dir())
            self.assertTrue((output_dir / "global_step_80").is_dir())
