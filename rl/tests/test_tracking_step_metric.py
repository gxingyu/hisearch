"""Restored training steps must not be discarded by W&B's history counter."""

import unittest
from unittest.mock import Mock, patch

from verl.utils.tracking import Tracking


class TrackingStepMetricTest(unittest.TestCase):
    def make_tracker(self, wandb_config):
        client = Mock()
        with patch.dict("sys.modules", {"wandb": client}):
            tracker = Tracking("project", "experiment", ["wandb"], config={
                "trainer": {}, "wandb": {"entity": "entity", **wandb_config},
            })
        return tracker, client

    def test_restored_and_backfilled_steps_use_append_only_history(self):
        tracker, client = self.make_tracker({"step_metric": "training/global_step"})
        console = Mock()
        tracker.logger["console"] = console
        original = {"val/f1": 0.4}
        for step in [69, 0, 41]:
            tracker.log(original, step=step)
        self.assertEqual(original, {"val/f1": 0.4})
        self.assertEqual([c.kwargs for c in client.log.call_args_list], [
            {"data": {"val/f1": 0.4, "training/global_step": step}, "commit": True}
            for step in [69, 0, 41]
        ])
        console.log.assert_called_with(data=original, step=41)
        client.define_metric.assert_any_call("*", step_metric="training/global_step", step_sync=False)
        client.log.reset_mock()
        tracker.log(original, step=42, backend=["console"])
        client.log.assert_not_called()
        tracker.finish()

    def test_default_preserves_explicit_step_semantics(self):
        tracker, client = self.make_tracker({})
        tracker.log({"train/loss": 0.2}, step=41)
        client.log.assert_called_once_with(data={"train/loss": 0.2}, step=41)
        client.define_metric.assert_not_called()
        tracker.finish()


if __name__ == "__main__":
    unittest.main()
