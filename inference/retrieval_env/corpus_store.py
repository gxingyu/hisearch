from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


class CorpusStore:
    def __init__(self, corpus_path: str | Path, data_root: str | Path):
        self.path = Path(corpus_path).resolve()
        self.data_root = Path(data_root).resolve()
        self.manifest = json.loads((self.data_root / "corpus_manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("ids") != "sequential strings, checked on every row":
            raise ValueError("Candidate views require the verified sequential passage ID mapping")
        self.offsets = np.load(self.data_root / "corpus_offsets.npy", mmap_mode="r")
        if len(self.offsets) != self.manifest["count"] + 1 or int(self.offsets[0]) != 0:
            raise ValueError("Corpus offset count does not match the manifest")
        if self.path.stat().st_size != self.manifest["size"] or int(self.offsets[-1]) != self.manifest["size"]:
            raise ValueError("Corpus size does not match retrieval offsets")
        self.version = f"wiki18:{self.manifest['sha256']}"

    def verify_snapshot(self) -> None:
        stat = self.path.stat()
        recorded_path = Path(self.manifest["path"]).resolve()
        if self.path == recorded_path and stat.st_mtime_ns == self.manifest["mtime_ns"]:
            return
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.manifest["sha256"]:
            raise ValueError("Shell corpus and dense retrieval corpus have different SHA256 values")

    def read(self, passage_id: Any) -> tuple[bytes, dict[str, Any]]:
        identifier = str(passage_id)
        try:
            position = int(identifier)
        except ValueError as exc:
            raise ValueError(f"Invalid passage ID {identifier!r}") from exc
        if not 0 <= position < self.manifest["count"]:
            raise ValueError(f"Passage ID {identifier!r} is outside the corpus")
        start = int(self.offsets[position])
        size = int(self.offsets[position + 1]) - start
        with self.path.open("rb") as stream:
            stream.seek(start)
            raw = stream.read(size)
        document = json.loads(raw)
        if str(document.get("id")) != identifier:
            raise ValueError("Passage ID does not match its verified corpus row")
        return raw, document


@lru_cache(maxsize=4)
def get_corpus_store(corpus_path: str, data_root: str) -> CorpusStore:
    return CorpusStore(corpus_path, data_root)
