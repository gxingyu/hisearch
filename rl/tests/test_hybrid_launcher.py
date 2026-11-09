# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "The RL launchers require bash on Linux")
class HybridLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        repository = Path(__file__).resolve().parents[2]
        for filename in ("run_gspo.sh", "run_gspo_hybrid.sh"):
            shutil.copyfile(repository / filename, self.root / filename)
        environment = self.root / ".venv/bin"
        environment.mkdir(parents=True)
        (environment / "activate").write_text('export PATH="${PWD}/.venv/bin:${PATH}"\n')
        self._executable(
            environment / "python",
            """#!/usr/bin/bash
set -eu
if [[ "${1:-}" == "-c" ]]; then
  printf 'http\\n127.0.0.1\\n8000\\n'
elif [[ "${1:-}" == "-" ]]; then
  exec "${MOCK_REAL_PYTHON}" "$@"
elif [[ "$*" == *--health-only* ]]; then
  [[ -f "${MOCK_READY}" ]] || exit 2
  if [[ "${MOCK_LEGACY_SERVICE:-0}" == "1" ]]; then
    printf '{"status":"ok"}\\n'
  else
    printf '{"status":"ok","parallel_shards":true,"batch_size":32,"batch_wait_ms":10.0}\\n'
  fi
else
  printf '{"tools":["shell","dense_search"]}\\n'
fi
""",
        )
        self._executable(
            environment / "bash",
            """#!/usr/bin/bash
set -eu
if [[ "${1:-}" == "rl/run_rl.sh" ]]; then
  shift
  printf '%s\\n' "$@" >"${MOCK_RL_ARGS}"
  printf '%s\\n' \
    "GREPSEEK_EXPERIMENT_NAME=${GREPSEEK_EXPERIMENT_NAME:-}" \
    "GREPSEEK_OUTPUT_DIR=${GREPSEEK_OUTPUT_DIR:-}" \
    "GREPSEEK_MODEL_PATH=${GREPSEEK_MODEL_PATH:-}" \
    "WANDB_PROJECT=${WANDB_PROJECT:-}" \
    "WANDB_RUN_ID=${WANDB_RUN_ID:-}" \
    "WANDB_RESUME=${WANDB_RESUME:-}" >"${MOCK_RL_ENV}"
else
  exec /usr/bin/bash "$@"
fi
""",
        )
        self._executable(
            self.root / "run_search_r1.sh",
            """#!/usr/bin/bash
set -eu
touch "${MOCK_SERVICE_STARTED}"
printf '%s\\n' "$@" >"${MOCK_SERVICE_ARGS}"
if [[ "${MOCK_SERVICE_FAIL:-0}" == "1" ]]; then
  echo 'mock index loading failure' >&2
  exit 7
fi
touch "${MOCK_READY}"
exec sleep 60
""",
        )
        model = self.root / "models/sft"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        self.ready = self.root / "ready"
        self.capture = self.root / "rl_arguments"
        self.environment_capture = self.root / "rl_environment"
        self.started = self.root / "service_started"
        self.service_arguments = self.root / "service_arguments"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _executable(path, contents):
        path.write_text(contents)
        path.chmod(0o755)

    def run_script(self, filename, name, *arguments, **overrides):
        environment = {
            **os.environ,
            "GREPSEEK_EXPERIMENT_NAME": name,
            "GREPSEEK_OUTPUT_DIR": str(self.root / "checkpoints/rl" / name),
            "MOCK_READY": str(self.ready),
            "MOCK_RL_ARGS": str(self.capture),
            "MOCK_RL_ENV": str(self.environment_capture),
            "MOCK_REAL_PYTHON": sys.executable,
            "MOCK_SERVICE_STARTED": str(self.started),
            "MOCK_SERVICE_ARGS": str(self.service_arguments),
            "GREPSEEK_DENSE_STARTUP_TIMEOUT": "5",
            "GREPSEEK_DENSE_URL": "http://127.0.0.1:8000/retrieve",
            "GREPSEEK_DENSE_BATCH_SIZE": "32",
            "GREPSEEK_DENSE_BATCH_WAIT_MS": "10",
            "GREPSEEK_DENSE_QUEUE_SIZE": "4096",
            "GREPSEEK_EXPERIMENT_NAME_EXPLICIT": "1",
            "WANDB_PROJECT": "grepseek-rl",
            "WANDB_API_KEY": "test-key",
            **overrides,
        }
        return subprocess.run(
            ["/usr/bin/bash", str(self.root / filename), *arguments],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_hybrid_inherits_every_baseline_training_argument(self):
        self.ready.touch()
        baseline = self.run_script("run_gspo.sh", "baseline")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        baseline_arguments = self.capture.read_text().splitlines()
        hybrid = self.run_script("run_gspo_hybrid.sh", "hybrid")
        self.assertEqual(hybrid.returncode, 0, hybrid.stderr)
        hybrid_arguments = self.capture.read_text().splitlines()
        self.assertEqual(hybrid_arguments[: len(baseline_arguments)], baseline_arguments)
        self.assertEqual(len(hybrid_arguments) - len(baseline_arguments), 7)
        self.assertIn("data.custom_cls.name=HybridDataset", hybrid_arguments)
        self.assertIn("actor_rollout_ref.rollout.agent.default_agent_loop=grepseek_hybrid_agent", hybrid_arguments)
        self.assertFalse(self.started.exists())

    def test_missing_service_starts_and_owned_process_is_cleaned_up(self):
        result = self.run_script("run_gspo_hybrid.sh", "started")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.started.exists())
        pid = int((self.root / "outputs/hybrid_startup/started/dense_server.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertTrue(self.capture.exists())
        service_arguments = self.service_arguments.read_text().splitlines()
        for option, value in (("--batch-size", "32"), ("--batch-wait-ms", "10"), ("--queue-size", "4096")):
            self.assertEqual(service_arguments[service_arguments.index(option) + 1], value)

    def test_default_launch_ignores_stale_experiment_and_starts_from_sft(self):
        self.ready.touch()
        (self.root / "checkpoints/rl/stale").mkdir(parents=True)
        result = self.run_script(
            "run_gspo_hybrid.sh",
            "stale",
            GREPSEEK_EXPERIMENT_NAME_EXPLICIT="0",
            WANDB_RUN_ID="old-run-id",
            WANDB_RESUME="must",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        environment = dict(line.split("=", 1) for line in self.environment_capture.read_text().splitlines())
        name = environment["GREPSEEK_EXPERIMENT_NAME"]
        self.assertRegex(name, r"^hisearch-gspo-hybrid-\d{8}-\d{6}$")
        self.assertEqual(environment["GREPSEEK_OUTPUT_DIR"], str(self.root / "checkpoints/rl" / name))
        self.assertEqual(
            environment["GREPSEEK_MODEL_PATH"],
            str(self.root / "models/sft"),
        )
        self.assertEqual(environment["WANDB_PROJECT"], "grepseek-rl")
        self.assertEqual(environment["WANDB_RUN_ID"], "")
        self.assertEqual(environment["WANDB_RESUME"], "")
        self.assertIn("trainer.resume_mode=disable", self.capture.read_text().splitlines())
        self.assertIn(f"W&B experiment: {name}", result.stdout)
        self.assertFalse(self.started.exists())

    def test_existing_legacy_service_prevents_training(self):
        self.ready.touch()
        result = self.run_script("run_gspo_hybrid.sh", "legacy", MOCK_LEGACY_SERVICE="1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("optimized batched/parallel settings", result.stderr)
        self.assertFalse(self.started.exists())
        self.assertFalse(self.capture.exists())

    def test_service_failure_prevents_training(self):
        result = self.run_script("run_gspo_hybrid.sh", "failed", MOCK_SERVICE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mock index loading failure", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_check_only_does_not_start_service_or_training(self):
        result = self.run_script("run_gspo_hybrid.sh", "check", "--check-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.started.exists())
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
