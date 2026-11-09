from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from inference.retrieval_env.corpus_store import CorpusStore


class CandidateError(Exception):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status


@dataclass
class CandidateSet:
    path: Path
    passage_ids: list[str]
    created_at: float


class RetrievalSession:
    def __init__(self, request_id: str, store: CorpusStore, ttl_seconds: float = 3600.0, tmp_root: str | None = None):
        self.request_id = request_id
        self.store = store
        self.ttl_seconds = ttl_seconds
        self._temporary = tempfile.TemporaryDirectory(prefix="grepseek_candidates_", dir=tmp_root)
        self.candidate_sets: dict[str, CandidateSet] = {}
        self.closed = False

    def register(self, results: list[dict[str, Any]]) -> str:
        if self.closed:
            raise CandidateError("invalid_candidate_set", "Retrieval session has ended")
        records = []
        identifiers = []
        for result in results:
            document = result["document"]
            identifier = str(document["id"])
            try:
                raw, original = self.store.read(identifier)
            except ValueError as exc:
                raise CandidateError("corpus_mismatch", str(exc)) from exc
            if document["contents"] != original.get("contents"):
                raise CandidateError("corpus_mismatch", "Dense passage does not match the original shell corpus")
            records.append(raw)
            identifiers.append(identifier)
        handle = uuid4().hex
        directory = Path(self._temporary.name) / handle
        directory.mkdir()
        path = directory / "corpus.jsonl"
        path.write_bytes(b"".join(records))
        self.candidate_sets[handle] = CandidateSet(path, identifiers, time.monotonic())
        return handle

    def resolve(self, handle: str, request_id: str) -> CandidateSet:
        if self.closed or request_id != self.request_id or handle not in self.candidate_sets:
            raise CandidateError("invalid_candidate_set", "Candidate handle does not belong to this rollout")
        candidates = self.candidate_sets[handle]
        if time.monotonic() - candidates.created_at > self.ttl_seconds:
            candidates.path.unlink(missing_ok=True)
            del self.candidate_sets[handle]
            raise CandidateError("expired_candidate_set", "Candidate handle has expired")
        return candidates

    def close(self) -> None:
        self.closed = True
        self.candidate_sets.clear()
        self._temporary.cleanup()
