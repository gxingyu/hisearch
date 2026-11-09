from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

MODULE = Path(__file__).resolve().parents[1] / "filter_redundant_trajectories.py"
spec = importlib.util.spec_from_file_location("redundancy_filter", MODULE)
filt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filt)


def example():
    return {"id": "a", "question_id": "q", "messages": [
        {"role": "user", "content": "Who owns the team?"},
        {"role": "assistant", "content": '<think>Find the owner.</think><tool_call>{"name":"dense_search","arguments":{"query":"team owner"}}</tool_call>'},
        {"role": "tool", "content": '{"contents":"OBSERVATION_ONLY_SECRET"}'},
        {"role": "assistant", "content": '<think>Ada owns the team, so I will confirm.</think><tool_call>{"name":"shell","arguments":{"command":"QUERY_ONLY_SECRET"}}</tool_call>'},
        {"role": "tool", "content": '{"contents":"FUTURE_OBSERVATION_SECRET"}'},
        {"role": "assistant", "content": '<think>FINAL_REASONING_SECRET</think><answer>FINAL_ANSWER_SECRET</answer>'},
    ], "tools": []}


class RedundancyFilterTests(unittest.TestCase):
    def test_prefixes_never_expose_observations_queries_or_final_answer(self):
        row = example()
        items = filt.prefixes(row)
        self.assertEqual(len(items), 2)
        first, second = map(json.dumps, items)
        self.assertNotIn("Ada", first)
        self.assertIn("Ada", second)
        for secret in ("OBSERVATION_ONLY_SECRET", "QUERY_ONLY_SECRET", "FUTURE_OBSERVATION_SECRET", "FINAL_REASONING_SECRET", "FINAL_ANSWER_SECRET"):
            self.assertNotIn(secret, first + second)

    def test_ready_requires_quote_from_correct_past_reasoning_turn(self):
        prefix = filt.prefixes(example())[1]
        result = {"status": "ready", "answer": "Ada", "source_turn": 2,
                  "reasoning_quote": "Ada owns the team", "explanation": "Complete answer before confirmation."}
        self.assertEqual(filt.validate_result(result, prefix), result)
        for patch in ({"source_turn": 3}, {"source_turn": 1}, {"reasoning_quote": "Invented conclusion"}, {"answer": ""}):
            with self.assertRaises(ValueError):
                filt.validate_result(result | patch, prefix)

    def test_uncertainty_is_distinct_from_ready_and_schema_is_strict(self):
        prefix = filt.prefixes(example())[0]
        result = {"status": "uncertain", "answer": "", "source_turn": 0,
                  "reasoning_quote": "", "explanation": "Unresolved."}
        self.assertEqual(filt.validate_result(result, prefix), result)
        with self.assertRaises(ValueError):
            filt.validate_result(result | {"answer": "Ada"}, prefix)
        with self.assertRaises(ValueError):
            filt.validate_result(result | {"extra": True}, prefix)

    def test_malformed_unpaired_calls_fail_instead_of_being_kept(self):
        row = example()
        del row["messages"][2]
        with self.assertRaises(ValueError):
            filt.prefixes(row)

    def test_export_is_partitioned_and_preserves_full_rows_and_split(self):
        train = [example(), example(), example()]
        for index, row in enumerate(train):
            row["id"], row["question_id"] = f"t{index}", f"q{index}"
        val = [example()]
        val[0]["id"], val[0]["question_id"] = "v0", "qv"
        tables = {"train": pa.Table.from_pylist(train), "val": pa.Table.from_pylist(val)}
        records = [{"id": row["id"], "question_id": row["question_id"], "split": split, "row_index": index}
                   for split, rows in (("train", train), ("val", val)) for index, row in enumerate(rows)]
        metadata = {r["id"]: copy.deepcopy(r) for r in train + val}
        categories = {"t0": "kept", "t1": "removed", "t2": "review_required", "v0": "kept"}
        results = {rid: {"category": cat} for rid, cat in categories.items()}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(out_dir=Path(directory))
            summary = filt.export(args, tables, records, metadata, results)
            self.assertEqual(summary["counts"], {"kept": 2, "removed": 1, "review_required": 1})
            self.assertEqual(pq.read_table(Path(directory)/"kept/parquet/train.parquet").to_pylist(), [train[0]])
            self.assertEqual(pq.read_table(Path(directory)/"kept/parquet/val.parquet").to_pylist(), val)
            with self.assertRaises(ValueError):
                filt.export(args, tables, records, metadata, {"t0": results["t0"]})


if __name__ == "__main__":
    unittest.main()
