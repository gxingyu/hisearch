from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from inference.parallel_search.subprocesses import run_pipeline


class PipelineProcessTests(unittest.TestCase):
    def test_output_limit_stops_an_unbounded_producer(self):
        result = run_pipeline("yes passage", timeout=2, max_output_bytes=4096)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.stdout.encode()), 4096)
        self.assertEqual(result.returncode, 0)

    def test_large_stdin_and_stdout_do_not_deadlock(self):
        contents = "passage\n" * 100000
        result = run_pipeline("cat", timeout=5, input=contents, max_output_bytes=0)
        self.assertEqual(result.stdout, contents)
        self.assertFalse(result.truncated)

    def test_timeout_kills_background_children(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            script = "import time; time.sleep(60)"
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} & echo $! > {shlex.quote(str(pid_file))}; wait"
            with self.assertRaises(subprocess.TimeoutExpired):
                run_pipeline(command, timeout=0.3)
            child_pid = int(pid_file.read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status_file = Path(f"/proc/{child_pid}/stat")
                if not status_file.exists() or status_file.read_text().split()[2] == "Z":
                    break
                time.sleep(0.01)
            else:
                os.kill(child_pid, 9)
                self.fail("Timed out pipeline left its child running")
