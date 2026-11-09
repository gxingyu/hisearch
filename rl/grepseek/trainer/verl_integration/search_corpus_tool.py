# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
"""VERL BaseTool that mirrors the cold-start data-gen `shell` tool exactly.

Implementation parity with `sft/data_generation/utils/tools.py`:
  - SFT validator (whitelist + dangerous-pattern blocklist) with an RL guard
    against loading the corpus as a ripgrep pattern file
  - same `corpus.jsonl -> wiki_corpus.jsonl` rewriting before execution
  - legacy subprocess execution (bash, cwd=corpus_root, capture stdout/stderr)
    or the opt-in sharded daemon backend used by RL rollout
  - same output truncation behavior

The RL payload keeps the authoritative stdout and execution metadata but omits
the derived `information_lines` list. That list duplicated stdout byte-for-byte
inside every model-visible observation and inflated rollout contexts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
from typing import Any, Optional
from uuid import uuid4

from inference.parallel_search.subprocesses import OUTPUT_TRUNCATION_MARKER, run_pipeline
from inference.shell_syntax import split_pipeline
from grepseek.trainer.verl_integration.tool_executors import get_tool_executor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _truncate_tokens(text: str, max_tokens: int, tokenizer) -> tuple[str, bool]:
    """Truncate `text` to at most `max_tokens` tokens. Returns (text, was_truncated).
    Tokenizes an expanding prefix so large tool outputs do not require a full
    encoding. The tokenizer is injected by the agent loop."""
    if not text:
        return text, False
    try:
        prefix_length = min(len(text), max(256, max_tokens * 8))
        while True:
            ids = tokenizer(text[:prefix_length], add_special_tokens=False)["input_ids"]
            if prefix_length == len(text):
                if len(ids) <= max_tokens:
                    return text, False
                return tokenizer.decode(ids[:max_tokens], skip_special_tokens=False), True
            if len(ids) > max_tokens + 32:
                return tokenizer.decode(ids[:max_tokens], skip_special_tokens=False), True
            prefix_length = min(len(text), prefix_length * 2)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.warning(
            "Tokenizer failed while truncating a tool response; using character fallback: %s",
            exc,
        )
        return text[:max_tokens], len(text) > max_tokens


# ---------------------------------------------------------------------------
# Validator + executor — verbatim copies of sft/data_generation/utils/tools.py
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = {
    "rg", "grep", "find", "sed", "awk", "head", "tail", "cat",
    "ls", "wc", "sort", "cut", "uniq", "tr",
}

CORPUS_FILENAME_LOGICAL = "corpus.jsonl"      # what the agent types
CORPUS_FILENAME_ACTUAL = "wiki_corpus.jsonl"  # what's on disk in corpus_root


def _augment_rg_flags(cmd: str) -> str:
    """Add deterministic fast-path flags to each ripgrep stage."""
    stages = split_pipeline(cmd)
    rewritten: list[str] = []
    for raw_stage in stages:
        stage = raw_stage.strip()
        try:
            tokens = shlex.split(stage)
        except ValueError:
            rewritten.append(raw_stage)
            continue
        if not tokens or tokens[0].split("/")[-1] != "rg":
            rewritten.append(raw_stage)
            continue
        flags = [flag for flag in ("--mmap", "--no-config") if flag not in tokens]
        if not flags:
            rewritten.append(raw_stage)
            continue
        program = tokens[0]
        insert_at = stage.find(program) + len(program)
        rewritten.append(stage[:insert_at] + " " + " ".join(flags) + stage[insert_at:])
    return "|".join(rewritten)


def _search_env() -> dict[str, str]:
    env = {**os.environ, "LC_ALL": "C"}
    env.pop("LD_LIBRARY_PATH", None)
    return env


class ToolError(Exception):
    pass


def _strip_quoted(s: str) -> str:
    s = re.sub(r"'[^']*'", "", s)
    s = re.sub(r'"[^"]*"', "", s)
    return s


_DANGEROUS_PATTERNS = [
    (re.compile(r"\bsed\b[^|]*\s-(?:[a-zA-Z]*i|-in-place)\b"), "sed -i / --in-place (in-place edit)"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s-i\s+inplace\b"), "awk/gawk -i inplace"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s--in-place\b"), "awk --in-place"),
    (re.compile(r"\bfind\b[^|]*\s-delete\b"), "find -delete"),
    (re.compile(r"\bfind\b[^|]*\s-exec(?:dir)?\b"), "find -exec / -execdir"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bsystem\s*\("), "awk system() call"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bprintf?\b[^|]*>\s*[\"']"), "awk script redirection (print > file)"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\|\s*[\"']"), "awk script pipe-to-shell-command"),
    (re.compile(r"\bsed\b[^|]*'\s*e\s+"), "sed 'e' (execute shell) command"),
    (re.compile(r"\b(?:rm|mv|cp|dd|chmod|chown|chgrp|truncate|install|tee|ln)\b"),
     "destructive/write command name"),
]


def _validate_pipeline(cmd: str) -> None:
    if "\n" in cmd:
        raise ToolError("newlines not allowed in command")
    if len(cmd) > 2000:
        raise ToolError("command too long")
    for pat, label in _DANGEROUS_PATTERNS:
        if pat.search(cmd):
            raise ToolError(f"disallowed: {label}")
    bare = _strip_quoted(cmd)
    for bad in ("`", "$(", "&&", "||", ";", ">", "<"):
        if bad in bare:
            raise ToolError(f"disallowed shell construct: {bad!r}")
    segments = [s.strip() for s in split_pipeline(cmd)]
    for seg in segments:
        if not seg:
            raise ToolError("empty pipeline segment")
        try:
            tokens = shlex.split(seg)
        except ValueError as e:
            raise ToolError(f"could not parse segment: {e}")
        if not tokens:
            raise ToolError("empty command")
        prog = tokens[0].split("/")[-1]
        if prog not in ALLOWED_COMMANDS:
            raise ToolError(f"command '{prog}' not in whitelist")
        if prog in ("rg", "grep"):
            for token in tokens[1:]:
                if token == "--":
                    break
                if (token in ("-f", "--file") or token.startswith("--file=")
                        or (token.startswith("-") and not token.startswith("--")
                            and "f" in token[1:])):
                    raise ToolError(
                        "rg/grep -f/--file loads patterns from a file; "
                        "use a quoted pattern and corpus.jsonl as the input file"
                    )


def _execute_pipeline(
    cmd: str,
    *,
    cwd: str,
    timeout: float,
    max_tokens: int,
    tokenizer,
    engine=None,
) -> dict:
    """Run a validated pipeline. Returns the JSON-shaped payload the SFT pipeline
    emits in the trajectory's tool message (see
    sft/data_generation/utils/pipeline.py:_verl_style_tool_message_content).

    Output is truncated to `max_tokens` tokens using the same tokenizer the
    trained model uses. A character limit would over-cut some queries and
    under-cut others."""
    # Rewrite corpus.jsonl -> wiki_corpus.jsonl, exactly like the SFT pipeline.
    cmd_to_run = cmd.replace(CORPUS_FILENAME_LOGICAL, CORPUS_FILENAME_ACTUAL)
    if engine is not None:
        cmd_to_run = _augment_rg_flags(cmd_to_run)
    timed_out = False
    if engine is not None:
        stdout, exit_code = engine.execute(cmd_to_run, env=_search_env())
        stderr = ""
        if exit_code == -1:
            stdout = f"[command timed out after {timeout}s]"
            timed_out = True
    else:
        try:
            proc = run_pipeline(
                cmd_to_run,
                cwd=cwd,
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            if proc.truncated:
                stdout += OUTPUT_TRUNCATION_MARKER
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            stdout = f"[command timed out after {timeout}s]"
            stderr = ""
            exit_code = -1
            timed_out = True

    source_truncated = OUTPUT_TRUNCATION_MARKER in stdout
    stdout = stdout.replace(OUTPUT_TRUNCATION_MARKER, "")
    stdout, truncated = _truncate_tokens(stdout, max_tokens, tokenizer)
    truncated = truncated or source_truncated
    if truncated:
        stdout += f"\n[... output truncated at {max_tokens} tokens]"

    return {
        "command": cmd,                # report the AGENT's command, not the rewritten one
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# VERL BaseTool wrapper
# ---------------------------------------------------------------------------


class SearchCorpusTool:
    """VERL tool for shell pipelines against the local Wikipedia corpus."""

    def __init__(self, config: dict[str, Any], tool_schema) -> None:
        # Env var takes precedence over YAML so launchers can redirect to a
        # local corpus copy without editing the tool config file. run_rl.sh sets
        # VERL_TOOL_CORPUS_ROOT from GREPSEEK_CORPUS_ROOT (point it at a local-disk
        # copy for rg speed). Falls back to the YAML corpus_root when unset.
        env_corpus_root = os.environ.get("VERL_TOOL_CORPUS_ROOT", "").strip()
        corpus_root = env_corpus_root or config.get("corpus_root")
        if not corpus_root:
            raise ValueError("SearchCorpusTool requires 'corpus_root' in config "
                             "or VERL_TOOL_CORPUS_ROOT env var")
        self.corpus_root = str(corpus_root)
        # Hard check the file is actually there — fail fast at tool-init rather
        # than producing empty stdout on every rollout.
        _corpus_file = os.path.join(self.corpus_root, "wiki_corpus.jsonl")
        if not os.path.isfile(_corpus_file):
            raise ValueError(
                f"SearchCorpusTool: corpus file not found at {_corpus_file!r}. "
                f"corpus_root={self.corpus_root!r} "
                f"(VERL_TOOL_CORPUS_ROOT={env_corpus_root!r}, "
                f"yaml corpus_root={config.get('corpus_root')!r})"
            )
        self.timeout_seconds: float = float(config.get("timeout_seconds", 180.0))
        self.max_output_tokens: int = int(config.get("max_output_tokens", 1024))
        self._executor = get_tool_executor("shell", int(config.get("executor_workers", 16)))
        self.engine = self._build_search_engine(config, self.corpus_root)

        # Tokenizer is INJECTED by GrepSeekAgentLoop.__init__ after tool
        # construction (see set_tokenizer below) — it uses the same tokenizer
        # the agent loop loaded from actor_rollout_ref.model.path. Zero config
        # duplication: whatever model is being trained, that tokenizer applies
        # the configured RL observation limit.
        self._tokenizer = None

        self.config = config
        self.tool_schema = tool_schema
        self.name: str = tool_schema.function.name if tool_schema else "shell"
        self._instances: dict[str, dict] = {}

    @staticmethod
    def _build_search_engine(config: dict[str, Any], corpus_root: str):
        mode = os.environ.get("GREPSEEK_SEARCH_ENGINE", "none").strip().lower()
        if mode in ("", "none", "off"):
            return None
        if mode == "daemon":
            from inference.parallel_search import DaemonClient

            socket_path = os.environ.get("GREPSEEK_SEARCH_SOCKET", "").strip()
            if not socket_path:
                raise ValueError(
                    "GREPSEEK_SEARCH_ENGINE=daemon requires "
                    "GREPSEEK_SEARCH_SOCKET"
                )
            return DaemonClient(
                socket_path=socket_path,
                timeout=float(config.get("timeout_seconds", 180.0)) + 10.0,
            )
        if mode == "inproc":
            from inference.parallel_search import ShardedSearchEngine

            shard_dir = os.environ.get("GREPSEEK_SEARCH_SHARD_DIR", "").strip()
            if not shard_dir:
                raise ValueError(
                    "GREPSEEK_SEARCH_ENGINE=inproc requires "
                    "GREPSEEK_SEARCH_SHARD_DIR"
                )
            return ShardedSearchEngine(
                shard_dir=shard_dir,
                fallback_corpus_file=os.path.join(corpus_root, "wiki_corpus.jsonl"),
                timeout=float(config.get("timeout_seconds", 180.0)),
            )
        raise ValueError(f"unknown GREPSEEK_SEARCH_ENGINE={mode!r}")

    def set_tokenizer(self, tokenizer) -> None:
        """Called once by GrepSeekAgentLoop.__init__ to bind the trained
        model's tokenizer for token-based stdout truncation."""
        self._tokenizer = tokenizer

    def get_openai_tool_schema(self):
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, Any]:
        from verl.tools.schemas import ToolResponse

        iid = instance_id if instance_id is not None else str(uuid4())
        self._instances[iid] = {}
        return iid, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[Any, float, dict]:
        from verl.tools.schemas import ToolResponse

        command = (parameters.get("command") or "").strip()
        if not command:
            return ToolResponse(text="Error: empty command"), 0.0, {}

        # Validate (synchronous — pure string checks).
        try:
            _validate_pipeline(command)
        except ToolError as exc:
            err_payload = json.dumps(
                {
                    "command": command,
                    "stdout": "",
                    "stderr": f"validation error: {exc}",
                    "exit_code": -2,
                    "timed_out": False,
                    "truncated": False,
                },
                ensure_ascii=False,
            )
            return ToolResponse(text=err_payload), 0.0, {
                "validation_error": str(exc),
                "exit_code": -2,
                "timed_out": False,
                "status": "validation_error",
            }

        if self._tokenizer is None:
            raise RuntimeError(
                "SearchCorpusTool was invoked before set_tokenizer() was "
                "called. GrepSeekAgentLoop.__init__ should have injected "
                "the tokenizer from self.tokenizer; check that the agent loop "
                "is initialized correctly."
            )

        # Execute the pipeline in a thread to avoid blocking the asyncio event loop.
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            self._executor,
            lambda: _execute_pipeline(
                command,
                cwd=self.corpus_root,
                timeout=self.timeout_seconds,
                max_tokens=self.max_output_tokens,
                tokenizer=self._tokenizer,
                engine=self.engine,
            ),
        )

        response_text = json.dumps(payload, ensure_ascii=False)
        metrics = {
            "exit_code": payload["exit_code"],
            "timed_out": payload["timed_out"],
            "truncated": payload["truncated"],
        }
        return ToolResponse(text=response_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instances.pop(instance_id, None)
