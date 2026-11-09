from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]

from inference.retrieval_env.schemas import tool_schemas
from to_parquet import in_val


class HybridExportTests(unittest.TestCase):
    def test_parquet_keeps_strategies_together_and_both_tools(self):
        train_id = next(f"q{index}" for index in range(100) if not in_val(f"q{index}", 0.5))
        val_id = next(f"q{index}" for index in range(100) if in_val(f"q{index}", 0.5))
        rows = [
            {"id": question_id, "question_id": question_id,
             "trajectory_id": f"{question_id}::{strategy}::0", "tools": tool_schemas(),
             "messages": [{"role": "user", "content": "A question"},
                          {"role": "assistant", "content": "An answer"}]}
            for question_id in (train_id, val_id)
            for strategy in ("dense_first", "shell_first")
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sft.jsonl"
            input_path.write_text("\n".join(json.dumps(row) for row in rows + rows[:1]), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "sft/data_generation/to_parquet.py"),
                 "--in", str(input_path), "--out_dir", directory, "--val_frac", "0.5", "--include_tools"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            train = parquet.read_table(Path(directory) / "train.parquet").to_pylist()
            val = parquet.read_table(Path(directory) / "val.parquet").to_pylist()
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        self.assertEqual({row["question_id"] for row in train}, {train_id})
        self.assertEqual({row["question_id"] for row in val}, {val_id})
        for row in train + val:
            self.assertEqual({tool["function"]["name"] for tool in row["tools"]}, {"shell", "dense_search"})


if __name__ == "__main__":
    unittest.main()
