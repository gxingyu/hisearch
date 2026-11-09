from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sft/data_generation")]

from inference.retrieval_env.corpus_store import CorpusStore
from utils.hybrid_tools import HybridToolEnvironment


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


class FixtureClient:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query):
        return [{"document": document, "score": 1.0, "rank": index + 1}
                for index, document in enumerate(self.documents)]


class HybridToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.documents = [
            {"id": "0", "contents": '"Ada"\nAda was born in London.'},
            {"id": "1", "contents": '"Charles"\nCharles was born in London.'},
        ]
        records = [(json.dumps(document) + "\n").encode() for document in self.documents]
        corpus = self.root / "wiki_corpus.jsonl"
        corpus.write_bytes(b"".join(records))
        np.save(self.root / "corpus_offsets.npy", np.array([0, len(records[0]), sum(map(len, records))], dtype=np.uint64))
        manifest = {
            "path": str(corpus), "mtime_ns": corpus.stat().st_mtime_ns,
            "size": corpus.stat().st_size, "count": 2,
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "ids": "sequential strings, checked on every row",
        }
        (self.root / "corpus_manifest.json").write_text(json.dumps(manifest))
        store = CorpusStore(corpus, self.root)
        with patch.dict(os.environ, {"GREPSEEK_SEARCH_ENGINE": "none"}):
            self.environment = HybridToolEnvironment(store, FixtureClient(self.documents[:1]), CharacterTokenizer(), 2048)
        self.session = self.environment.session("test")

    def tearDown(self):
        self.session.close()
        self.temporary.cleanup()

    def execute(self, name, arguments):
        return self.environment.execute({"name": name, "arguments": arguments}, self.session, 2)[0]

    def test_dense_candidate_isolation_and_corpus_fallback(self):
        dense = self.execute("dense_search", {"query": "Ada"})
        self.assertEqual(dense["returned_k"], 1)
        scope = {"type": "candidates", "candidate_set_id": dense["candidate_set_id"]}
        candidate = self.execute("shell", {"command": 'rg -F "Charles" corpus.jsonl', "scope": scope})
        self.assertEqual(candidate["status"], "no_hits")
        corpus = self.execute("shell", {"command": 'rg -F "Charles" corpus.jsonl | head -n 1'})
        self.assertEqual(corpus["status"], "ok")
        self.assertIn("Charles", corpus["stdout"])
        other = self.environment.session("other")
        try:
            payload = self.environment.execute(
                {"name": "shell", "arguments": {"command": "cat corpus.jsonl", "scope": scope}}, other, 2,
            )[0]
            self.assertEqual(payload["status"], "invalid_candidate_set")
        finally:
            other.close()

    def test_failure_metadata_is_real(self):
        failure = self.execute("shell", {"command": 'rg "[" corpus.jsonl'})
        self.assertEqual(failure["status"], "execution_error")
        self.assertNotEqual(failure["exit_code"], 0)
        self.assertTrue(failure["stderr"])
        missing = self.execute("shell", {"command": 'rg -F "missing" corpus.jsonl'})
        self.assertEqual(missing["status"], "no_hits")
        self.assertEqual(missing["exit_code"], 1)

    def test_branch_clones_candidate_handles_and_full_raw_content(self):
        dense = self.execute("dense_search", {"query": "Ada"})
        old_handle = dense["candidate_set_id"]
        dense["documents"][0]["contents"] = "Ada"
        dense["documents"][0]["content_truncated"] = True
        prefix = [{"think": "Search Ada.", "action": {"name": "dense_search", "arguments": {"query": "Ada"}},
                   "observation": dense, "metrics": {"elapsed_seconds": 0}}]
        other = self.environment.session("branch")
        try:
            cloned = self.environment.clone_prefix(prefix, other)
            new_handle = cloned[0]["observation"]["candidate_set_id"]
            self.assertNotEqual(old_handle, new_handle)
            self.assertEqual(cloned[0]["observation"]["documents"][0]["contents"], "Ada")
            payload = self.environment.execute({"name": "shell", "arguments": {
                "command": "cat corpus.jsonl", "scope": {"type": "candidates", "candidate_set_id": new_handle}}}, other, 1)[0]
            self.assertIn("London", payload["stdout"])
            self.assertEqual(prefix[0]["observation"]["candidate_set_id"], old_handle)
        finally:
            other.close()

    def test_invalid_schema_and_write_command(self):
        invalid = self.execute("dense_search", {"query": "Ada", "k": 10})
        self.assertEqual(invalid["status"], "invalid_request")
        destructive = self.execute("shell", {"command": "sed -i s/Ada/Bad/ corpus.jsonl"})
        self.assertEqual(destructive["status"], "invalid_request")
        self.assertIn("Ada", self.environment.store.read("0")[1]["contents"])


if __name__ == "__main__":
    unittest.main()
