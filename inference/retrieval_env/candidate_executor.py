from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from inference.retrieval_env.sessions import CandidateError

_FLAGS = {
    "rg": (set("FiwnvoclsHxEqUP"), {"e": "pattern", "m": "number", "A": "number", "B": "number", "C": "number"}),
    "grep": (set("FiwnvoclsHxEqP"), {"e": "pattern", "m": "number", "A": "number", "B": "number", "C": "number"}),
    "cat": (set("nbsvETA"), {}),
    "head": (set("qv"), {"n": "number", "c": "number"}),
    "tail": (set("qv"), {"n": "number", "c": "number"}),
    "wc": (set("lwcmL"), {}),
    "sort": (set("rnghfbdMVsum"), {"k": "key", "t": "delimiter"}),
    "cut": (set("s"), {"d": "delimiter", "f": "fields", "c": "fields", "b": "fields"}),
    "uniq": (set("cduiz"), {"f": "number", "s": "number", "w": "number"}),
    "tr": (set("cCdst"), {}),
}
_LONG_FLAGS = {
    "rg": {
        "fixed-strings",
        "ignore-case",
        "word-regexp",
        "line-number",
        "invert-match",
        "only-matching",
        "count",
        "no-heading",
        "no-config",
        "mmap",
    },
    "grep": {"fixed-strings", "ignore-case", "word-regexp", "line-number", "invert-match", "only-matching", "count"},
    "cat": {"number", "number-nonblank", "squeeze-blank"},
    "head": {"quiet", "verbose"},
    "tail": {"quiet", "verbose"},
    "wc": {"lines", "words", "bytes", "chars", "max-line-length"},
    "sort": {
        "reverse",
        "numeric-sort",
        "general-numeric-sort",
        "human-numeric-sort",
        "ignore-case",
        "unique",
        "stable",
    },
    "cut": {"complement", "only-delimited"},
    "uniq": {"count", "repeated", "unique", "ignore-case"},
    "tr": {"complement", "delete", "squeeze-repeats", "truncate-set1"},
}
_LONG_VALUES = {
    "rg": {
        "regexp": "pattern",
        "max-count": "number",
        "after-context": "number",
        "before-context": "number",
        "context": "number",
    },
    "grep": {
        "regexp": "pattern",
        "max-count": "number",
        "after-context": "number",
        "before-context": "number",
        "context": "number",
    },
    "head": {"lines": "number", "bytes": "number"},
    "tail": {"lines": "number", "bytes": "number"},
    "sort": {"key": "key", "field-separator": "delimiter"},
    "cut": {
        "delimiter": "delimiter",
        "fields": "fields",
        "characters": "fields",
        "bytes": "fields",
        "output-delimiter": "pattern",
    },
}


def _validate_value(value: str, kind: str) -> None:
    if kind == "number" and not re.fullmatch(r"[+-]?\d+", value):
        raise CandidateError("invalid_scope_command", "Candidate numeric options must be integers")
    if kind == "delimiter" and len(value) != 1:
        raise CandidateError("invalid_scope_command", "Candidate delimiter must be one character")
    if kind in ("fields", "key") and not re.fullmatch(r"[0-9.,bdfgiMhnrV-]+", value):
        raise CandidateError("invalid_scope_command", "Unsupported candidate field/key expression")


def _stage(tokens: list[str], path: Path) -> tuple[list[str], bool]:
    program = tokens[0]
    if program not in _FLAGS:
        raise CandidateError(
            "invalid_scope_command",
            f"Candidate scope does not support {program!r}; use corpus scope for other commands",
        )
    flags, value_flags = _FLAGS[program]
    operands = []
    rewritten = [program]
    has_pattern_option = False
    position = 1
    options = True
    while position < len(tokens):
        token = tokens[position]
        if options and token == "--":
            rewritten.append(token)
            options = False
        elif options and token.startswith("--"):
            option, separator, attached = token[2:].partition("=")
            if option in _LONG_FLAGS[program] and not separator:
                rewritten.append(token)
            elif option in _LONG_VALUES.get(program, {}):
                kind = _LONG_VALUES[program][option]
                if separator:
                    value = attached
                    rewritten.append(token)
                else:
                    position += 1
                    if position >= len(tokens):
                        raise CandidateError("invalid_scope_command", f"Missing value for --{option}")
                    value = tokens[position]
                    rewritten.extend([token, value])
                _validate_value(value, kind)
                has_pattern_option |= option == "regexp"
            else:
                raise CandidateError("invalid_scope_command", f"Unsupported candidate option --{option}")
        elif options and token.startswith("-") and token != "-":
            if program in ("head", "tail") and re.fullmatch(r"-\d+", token):
                rewritten.append(token)
            else:
                characters = token[1:]
                offset = 0
                while offset < len(characters):
                    flag = characters[offset]
                    if flag in flags:
                        offset += 1
                        continue
                    if flag not in value_flags:
                        raise CandidateError("invalid_scope_command", f"Unsupported candidate option -{flag}")
                    attached = characters[offset + 1 :]
                    if attached:
                        value = attached
                    else:
                        position += 1
                        if position >= len(tokens):
                            raise CandidateError("invalid_scope_command", f"Missing value for -{flag}")
                        value = tokens[position]
                    _validate_value(value, value_flags[flag])
                    has_pattern_option |= flag == "e" and program in ("rg", "grep")
                    break
                rewritten.append(token)
                if offset < len(characters) and not attached:
                    rewritten.append(value)
        else:
            operands.append((len(rewritten), token))
            rewritten.append(token)
        position += 1
    if program in ("rg", "grep") and not has_pattern_option:
        if not operands:
            raise CandidateError("invalid_scope_command", "Candidate search requires a pattern")
        file_operands = operands[1:]
    elif program == "tr":
        if not 1 <= len(operands) <= 2:
            raise CandidateError("invalid_scope_command", "tr requires one or two character sets")
        file_operands = []
    else:
        file_operands = operands
    if program == "uniq" and len(file_operands) > 1:
        raise CandidateError("invalid_scope_command", "uniq output files are not allowed")
    reads_corpus = False
    for index, filename in file_operands:
        if filename == "corpus.jsonl":
            rewritten[index] = str(path)
            reads_corpus = True
        elif filename != "-":
            raise CandidateError("invalid_scope_command", "Candidate file arguments must be corpus.jsonl or stdin (-)")
    return rewritten, reads_corpus


def candidate_pipeline(command: str, path: Path) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    stages = [[]]
    try:
        for token in lexer:
            if token == "|":
                if not stages[-1]:
                    raise CandidateError("invalid_scope_command", "Empty pipeline stage")
                stages.append([])
            else:
                stages[-1].append(token)
    except ValueError as exc:
        raise CandidateError("invalid_scope_command", str(exc)) from exc
    if not stages[-1]:
        raise CandidateError("invalid_scope_command", "Empty pipeline stage")
    parsed = [_stage(tokens, path) for tokens in stages]
    if not parsed[0][1]:
        raise CandidateError(
            "invalid_scope_command", "First candidate pipeline stage must read corpus.jsonl explicitly"
        )
    return [arguments for arguments, _ in parsed]


def execute_candidate_pipeline(
    command: str, path: Path, timeout: float, max_tokens: int, tokenizer: Any,
    *, validator=None, truncate_tokens=None, search_env=None,
) -> dict[str, Any]:
    from inference.parallel_search.subprocesses import run_pipeline
    if validator is None or truncate_tokens is None or search_env is None:
        from grepseek.trainer.verl_integration.search_corpus_tool import _search_env, _truncate_tokens, _validate_pipeline
        validator = validator or _validate_pipeline
        truncate_tokens = truncate_tokens or _truncate_tokens
        search_env = search_env or _search_env
    validator(command)
    stages = candidate_pipeline(command, path)
    deadline = time.monotonic() + timeout
    output = b""
    stderr = []
    return_codes = []
    timed_out = False
    source_truncated = False
    for stage_index, arguments in enumerate(stages):
        try:
            result = run_pipeline(
                arguments,
                input=output,
                cwd=path.parent,
                timeout=max(0.001, deadline - time.monotonic()),
                env=search_env(),
                text=False,
                max_output_bytes=None if stage_index == len(stages) - 1 else 0,
            )
        except subprocess.TimeoutExpired:
            output = f"[command timed out after {timeout}s]".encode()
            return_codes.append(-1)
            timed_out = True
            break
        output = result.stdout
        source_truncated = result.truncated
        stderr.append(result.stderr.decode("utf-8", errors="replace"))
        return_codes.append(result.returncode)
    stdout, truncated = truncate_tokens(output.decode("utf-8", errors="replace"), max_tokens, tokenizer)
    truncated = truncated or source_truncated
    if truncated:
        stdout += f"\n[... output truncated at {max_tokens} tokens]"
    no_hits = (
        not output and not timed_out and stages[0][0] in ("rg", "grep") and all(code in (0, 1) for code in return_codes)
    )
    return {
        "command": command,
        "stdout": stdout,
        "stderr": "".join(stderr),
        "exit_code": return_codes[-1],
        "timed_out": timed_out,
        "truncated": truncated,
        "status": "timeout"
        if timed_out
        else "no_hits"
        if no_hits
        else "ok"
        if return_codes[-1] == 0
        else "execution_error",
    }
