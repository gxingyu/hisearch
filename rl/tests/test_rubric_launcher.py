from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RubricLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.repo=Path(__file__).resolve().parents[2]
        shutil.copyfile(self.repo/'run_gspo_hybrid_rubric.sh',self.root/'run_gspo_hybrid_rubric.sh')
        bin_dir=self.root/'.venv/bin';bin_dir.mkdir(parents=True)
        (bin_dir/'activate').write_text('export PATH="${PWD}/.venv/bin:${PATH}"\n')
        self.executable(bin_dir/'python','''#!/usr/bin/bash
set -eu
printf '%s\\n' "$@" >>"${CAPTURE_PYTHON}"
if [[ "${MOCK_PREFLIGHT_FAILURE:-0}" == "1" ]]; then exit 2; fi
printf '{"policy_training_started":false}\\n'
''')
        self.executable(bin_dir/'bash','''#!/usr/bin/bash
set -eu
printf '%s\\n' "$@" >"${CAPTURE_DOWNSTREAM}"
printf '%s\\n' "${GREPSEEK_MODEL_PATH}" "${GREPSEEK_EXPERIMENT_NAME}" >"${CAPTURE_ENV}"
''')

    @staticmethod
    def executable(path,contents):
        path.write_text(contents);path.chmod(0o755)

    def tearDown(self):self.temp.cleanup()

    def run_script(self,*args,**env):
        environment={**os.environ,'CAPTURE_PYTHON':str(self.root/'python.args'),
                     'CAPTURE_DOWNSTREAM':str(self.root/'downstream.args'),'CAPTURE_ENV':str(self.root/'env'),
                     'GREPSEEK_RUBRIC_CALIBRATION_PATH':str(self.root/'train-reference.jsonl'),
                     'GREPSEEK_EXPERIMENT_NAME':'rubric-test','GREPSEEK_OUTPUT_DIR':str(self.root/'run'),**env}
        return subprocess.run(['/usr/bin/bash',str(self.root/'run_gspo_hybrid_rubric.sh'),*args],
                              cwd=self.root,env=environment,capture_output=True,text=True,timeout=10)

    def test_no_arguments_and_check_only_do_not_reach_training(self):
        for args in ((),('--check-only',)):
            with self.subTest(args=args):
                result=self.run_script(*args)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertFalse((self.root/'downstream.args').exists())
                self.assertNotIn('--live',(self.root/'python.args').read_text())
        self.assertFalse((self.root/'checkpoints').exists())

    def test_explicit_train_forwards_recipe_without_executing_real_training(self):
        result=self.run_script('--train','algorithm.rubric.per_item_reward=0.7',
                               'custom_reward_function.reward_kwargs.enable_length_decay=true',
                               'actor_rollout_ref.rollout.n=7')
        self.assertEqual(result.returncode,0,result.stderr)
        args=(self.root/'downstream.args').read_text().splitlines()
        settings=dict(a.split('=',1) for a in args[1:] if '=' in a)
        self.assertTrue(args[0].endswith('/run_gspo_hybrid.sh'))
        self.assertEqual(settings['algorithm.rubric.per_item_reward'],'0.1')
        self.assertEqual(settings['algorithm.rubric.enabled'],'true')
        self.assertEqual(settings['custom_reward_function.reward_kwargs.enable_length_decay'],'false')
        self.assertEqual(settings['actor_rollout_ref.actor.policy_loss.loss_mode'],'gspo')
        self.assertEqual(settings['actor_rollout_ref.rollout.n'],'7')
        self.assertEqual(settings['data.train_batch_size'],'256')
        self.assertEqual(settings['trainer.resume_mode'],'disable')
        self.assertEqual(settings['algorithm.filter_groups.enable'],'true')
        self.assertIn('--require-calibration',(self.root/'python.args').read_text())
        self.assertIn('--live',(self.root/'python.args').read_text())

    def test_failed_preflight_never_reaches_downstream(self):
        result=self.run_script('--train',MOCK_PREFLIGHT_FAILURE='1')
        self.assertEqual(result.returncode,2)
        self.assertFalse((self.root/'downstream.args').exists())

    def test_self_test_only_invokes_tests(self):
        result=self.run_script('--self-test')
        self.assertEqual(result.returncode,0,result.stderr)
        args=(self.root/'python.args').read_text()
        self.assertIn('unittest',args)
        self.assertIn('test_rubric_integration',args)
        self.assertFalse((self.root/'downstream.args').exists())

    def test_unknown_mode_is_rejected(self):
        result=self.run_script('--typo')
        self.assertEqual(result.returncode,2)
        self.assertFalse((self.root/'downstream.args').exists())


if __name__=='__main__':unittest.main()
