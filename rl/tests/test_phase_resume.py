import unittest
from unittest.mock import patch

import torch

from verl.trainer.main_ppo import create_rl_dataset
from verl.utils.checkpoint.fsdp_checkpoint_manager import restore_learning_rate
from verl.utils.torch_functional import get_constant_schedule_with_warmup


class PhaseResumeTests(unittest.TestCase):
    def test_restores_phase_learning_rate_and_scheduler(self):
        for checkpoint_step in (10, 40):
            with self.subTest(checkpoint_step=checkpoint_step):
                parameter = torch.nn.Parameter(torch.tensor(1.0))
                old_optimizer = torch.optim.AdamW([parameter], lr=5e-7)
                old_scheduler = get_constant_schedule_with_warmup(old_optimizer, num_warmup_steps=20)
                for _ in range(checkpoint_step):
                    parameter.grad = torch.ones_like(parameter)
                    old_optimizer.step()
                    old_scheduler.step()

                resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
                optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-6)
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=20)
                optimizer.load_state_dict(old_optimizer.state_dict())
                scheduler.load_state_dict(old_scheduler.state_dict())

                restore_learning_rate(optimizer, scheduler, 1e-6)

                expected_rate = 1e-6 * min(checkpoint_step / 20, 1.0)
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_rate)
                self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_rate)
                self.assertAlmostEqual(scheduler.base_lrs[0], 1e-6)
                self.assertTrue(optimizer.state[resumed_parameter])

                resumed_parameter.grad = torch.ones_like(resumed_parameter)
                optimizer.step()
                scheduler.step()
                expected_next_rate = 1e-6 * min((checkpoint_step + 1) / 20, 1.0)
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected_next_rate)

    def test_phase_warmup_survives_checkpoint_restart(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
        for _ in range(100):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()

        restore_learning_rate(
            optimizer,
            scheduler,
            1e-6,
            warmup_start_step=100,
            warmup_steps=20,
            warmup_start_lr=5e-7,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-7)
        for step in range(101, 111):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-7 + (step - 100) * 2.5e-8)

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
        resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-6)
        resumed_scheduler = get_constant_schedule_with_warmup(resumed_optimizer, num_warmup_steps=0)
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        restore_learning_rate(
            resumed_optimizer,
            resumed_scheduler,
            1e-6,
            warmup_start_step=100,
            warmup_steps=20,
            warmup_start_lr=5e-7,
        )
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 7.5e-7)
        self.assertTrue(resumed_optimizer.state[resumed_parameter])
        for step in range(111, 121):
            resumed_parameter.grad = torch.ones_like(resumed_parameter)
            resumed_optimizer.step()
            resumed_scheduler.step()
            self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 5e-7 + (step - 100) * 2.5e-8)
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 1e-6)

    def test_validation_dataset_receives_validation_marker(self):
        class Dataset:
            def __init__(self, is_train=True, **kwargs):
                self.is_train = is_train

        with patch("verl.utils.dataset.rl_dataset.get_dataset_class", return_value=Dataset):
            dataset = create_rl_dataset([], {}, None, None, is_train=False)

        self.assertFalse(dataset.is_train)


if __name__ == "__main__":
    unittest.main()
