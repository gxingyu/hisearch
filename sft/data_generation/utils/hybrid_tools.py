"""Synchronous adapter around the repository's hybrid retrieval tools."""
from __future__ import annotations

import copy
import os
import shlex
import subprocess
import time
from pathlib import Path

from inference.retrieval_env.candidate_executor import execute_candidate_pipeline
from inference.retrieval_env.corpus_store import CorpusStore
from inference.retrieval_env.dense_client import DenseClient, DenseServiceError
from inference.retrieval_env.observations import dense_observation
from inference.retrieval_env.schemas import tool_schema, validate_parameters
from inference.retrieval_env.sessions import CandidateError, RetrievalSession
from inference.parallel_search.subprocesses import run_pipeline
from .tools import ToolError, _validate_pipeline


def truncate_tokens(text, max_tokens, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text, False
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=False), True


def search_env():
    environment = os.environ.copy()
    environment["RIPGREP_CONFIG_PATH"] = ""
    return environment


def validate_command(command):
    _validate_pipeline(command)
    for segment in command.split("|"):
        tokens = shlex.split(segment)
        if tokens[0].split("/")[-1] in ("rg", "grep"):
            for token in tokens[1:]:
                if token == "--":
                    break
                if token.startswith("--file") or token.startswith("-") and not token.startswith("--") and "f" in token[1:]:
                    raise ToolError("Pattern files are not allowed")


class HybridToolEnvironment:
    def __init__(self, store, client, tokenizer, max_tokens=1024, timeout=60):
        self.store = store
        self.client = client
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.engine = None
        if os.environ.get("GREPSEEK_SEARCH_ENGINE", "none") == "daemon":
            from inference.parallel_search import DaemonClient

            self.engine = DaemonClient(socket_path=os.environ["GREPSEEK_SEARCH_SOCKET"], timeout=timeout + 10)

    @classmethod
    def from_paths(cls, corpus_dir, data_root, dense_url, tokenizer, max_tokens=1024, timeout=60):
        store = CorpusStore(Path(corpus_dir) / "wiki_corpus.jsonl", data_root)
        store.verify_snapshot()
        client = DenseClient(dense_url, timeout)
        health = client.health()
        if health.get("corpus_sha256") != store.manifest["sha256"] or health.get("rows") != store.manifest["count"]:
            raise ValueError("dense service and shell corpus snapshots differ")
        return cls(store, client, tokenizer, max_tokens, timeout)

    def session(self, request_id):
        return RetrievalSession(request_id, self.store)

    def clone_prefix(self, steps, session):
        cloned = copy.deepcopy(steps)
        handles = {}
        for step in cloned:
            scope = step["action"]["arguments"].get("scope", {})
            if scope.get("type") == "candidates":
                scope["candidate_set_id"] = handles[scope["candidate_set_id"]]
            observation = step["observation"]
            if observation.get("tool") == "dense_search" and observation.get("candidate_set_id"):
                results = []
                for document in observation.get("documents", []):
                    _, original = self.store.read(document["passage_id"])
                    results.append({"document": original, "rank": document["rank"], "score": document["score"]})
                old_handle = observation["candidate_set_id"]
                handles[old_handle] = session.register(results)
                observation["candidate_set_id"] = handles[old_handle]
            step["metrics"]["shared_prefix"] = True
        return cloned

    def execute(self, action, session, remaining):
        started = time.monotonic()
        name, arguments = action["name"], action["arguments"]
        scope = arguments.get("scope", {"type": "corpus"}) if name == "shell" else None
        payload = {"tool": name, "status": "invalid_request", "corpus_version": self.store.version}
        try:
            validate_parameters(arguments, tool_schema(name)["function"]["parameters"])
            if name == "dense_search":
                results = self.client.retrieve(arguments["query"])
                handle = session.register(results)
                payload = dense_observation(
                    results, handle, self.store.version, self.tokenizer, self.max_tokens
                )
            else:
                validate_command(arguments["command"])
                if scope["type"] == "candidates":
                    candidates = session.resolve(scope["candidate_set_id"], session.request_id)
                    payload = execute_candidate_pipeline(
                        arguments["command"], candidates.path, self.timeout,
                        self.max_tokens, self.tokenizer,
                        validator=validate_command, truncate_tokens=truncate_tokens, search_env=search_env,
                    )
                    payload.update(
                        candidate_count=len(candidates.passage_ids),
                        source_passage_ids=candidates.passage_ids,
                    )
                else:
                    payload = self.execute_corpus(arguments["command"])
                payload.update(tool="shell", scope=scope, corpus_version=self.store.version)
                payload.setdefault(
                    "status",
                    "timeout" if payload.get("timed_out") else
                    "execution_error" if payload.get("exit_code") not in (0, 1) or payload.get("stderr") else
                    "no_hits" if not payload.get("stdout", "").strip() else "ok",
                )
        except (CandidateError, DenseServiceError) as error:
            payload.update(status=error.status, error=str(error))
        except (ValueError, ToolError) as error:
            payload.update(status="invalid_request", error=str(error))
        except Exception as error:
            payload.update(status="execution_error", error=str(error))
        payload["remaining_tool_calls"] = remaining
        metrics = {
            "tool": name, "scope": scope.get("type") if scope else None,
            "status": payload.get("status"), "elapsed_seconds": time.monotonic() - started,
        }
        return payload, metrics

    def execute_corpus(self, command):
        rewritten = command.replace("corpus.jsonl", "wiki_corpus.jsonl")
        try:
            if self.engine is not None:
                stdout, exit_code = self.engine.execute(rewritten, env=search_env())
                stderr, source_truncated = "", "output truncated at source byte limit" in stdout
            else:
                result = run_pipeline(
                    rewritten, cwd=str(self.store.path.parent), timeout=self.timeout,
                    env=search_env(), max_output_bytes=65536,
                )
                stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
                source_truncated = result.truncated
        except subprocess.TimeoutExpired:
            stdout, stderr, exit_code, source_truncated = "", "", -1, False
        stdout, truncated = truncate_tokens(stdout, self.max_tokens, self.tokenizer)
        truncated = truncated or source_truncated
        if truncated:
            stdout += f"\n[... output truncated at {self.max_tokens} tokens]"
        return {
            "command": command, "stdout": stdout, "stderr": stderr,
            "exit_code": exit_code, "timed_out": exit_code == -1, "truncated": truncated,
        }
