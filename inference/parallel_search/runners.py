# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
"""Per-strategy execution: run a Plan against pre-built shards and merge the
results so the output is byte-identical to running the same pipeline against
the un-sharded corpus.

All runners take:
    - plan: the Plan from pipeline.classify()
    - shard_files: ordered list of absolute paths to shard files
    - corpus_filename_in_cmd: the logical filename the original command
      references (e.g. "corpus.jsonl" or "wiki_corpus.jsonl"). We substitute
      the per-shard absolute path for this token in each subshell.
    - timeout, env

Returns (stdout, returncode). returncode != 0 only when ALL per-shard
subshells fail; partial failures are surfaced via stderr but the pipeline
keeps going (matches the single-file `set -o pipefail`-off default).
"""
from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from .pipeline import Plan, Stage, Strategy
from .subprocesses import OUTPUT_TRUNCATION_MARKER, run_pipeline


def _stage_text(stages: Iterable[Stage]) -> str:
    """Join stages back into a `|` pipeline using their original raw text.

    We use Stage.raw (the pre-shlex string) so quoting is preserved exactly
    as the model wrote it — no risk of shlex re-quote drift.
    """
    return " | ".join(s.raw for s in stages)


def _sub_corpus_token(pipeline_text: str, corpus_token: str, shard_path: str) -> str:
    """Replace the corpus filename in the pipeline with a per-shard absolute
    path. We do a literal substring replacement; the existing inference.tools
    validator already rejects newlines / multiple inputs / shell metachars in
    the command, so a naive replace is safe in this context.
    """
    return pipeline_text.replace(corpus_token, shard_path)


def _run_one(cmd: str, timeout: float, env: dict | None) -> tuple[str, str, int]:
    try:
        proc = run_pipeline(
            cmd,
            timeout=timeout,
            env=env,
        )
        stdout = proc.stdout or ""
        if proc.truncated:
            stdout += OUTPUT_TRUNCATION_MARKER
        return stdout, proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"[shard timed out after {timeout}s]\n", -1


def _run_parallel(per_shard_cmds: list[str], timeout: float, env: dict | None
                  ) -> list[tuple[str, str, int]]:
    """Run a per-shard command list in parallel, preserving input order in
    the returned list. Uses threads (not processes) because each underlying
    `subprocess.run` already forks — adding a process pool on top would just
    double-fork without speedup.
    """
    results: list[tuple[str, str, int] | None] = [None] * len(per_shard_cmds)
    # max_workers = len(per_shard_cmds): we want all shards in flight at once.
    with ThreadPoolExecutor(max_workers=max(1, len(per_shard_cmds))) as pool:
        futures = {pool.submit(_run_one, c, timeout, env): i
                   for i, c in enumerate(per_shard_cmds)}
        for fut in futures:
            i = futures[fut]
            results[i] = fut.result()
    # Cast away the Optional; ThreadPoolExecutor.future is always resolved here.
    return [r for r in results if r is not None]  # type: ignore[return-value]


@dataclass(frozen=True)
class BoundedPrefixResult:
    stdout: str
    returncode: int
    overflow: bool
    duration_s: float


def _run_one_bounded(
    cmd: str,
    timeout: float,
    env: dict | None,
    max_bytes: int,
) -> tuple[bytes, int, bool]:
    """Run one prefix scan while bounding captured output memory."""
    bounded_cmd = f"{cmd} | head -c {max_bytes + 1}"
    try:
        proc = run_pipeline(
            bounded_cmd,
            timeout=timeout,
            text=False,
            env=env,
            max_output_bytes=0,
        )
    except subprocess.TimeoutExpired:
        return b"", -1, False
    overflow = len(proc.stdout) > max_bytes
    return proc.stdout[:max_bytes], proc.returncode, overflow


def run_first_stage_bounded(
    first_stage: Stage,
    *,
    shard_files: list[str],
    corpus_token: str,
    timeout: float,
    env: dict | None,
    max_bytes: int,
) -> BoundedPrefixResult:
    """Scan a first ``rg`` stage once, retaining it only below ``max_bytes``.

    A per-shard cap prevents broad patterns from materializing an unbounded
    result in the daemon. Overflow is a signal to execute the original full
    pipeline instead, preserving command semantics.
    """
    started = time.monotonic()
    per_shard_limit = max(1, max_bytes // max(1, len(shard_files)))
    commands = [
        _sub_corpus_token(first_stage.raw, corpus_token, shard_file)
        for shard_file in shard_files
    ]
    with ThreadPoolExecutor(max_workers=max(1, len(commands))) as pool:
        futures = [
            pool.submit(_run_one_bounded, command, timeout, env, per_shard_limit)
            for command in commands
        ]
        parts = [future.result() for future in futures]

    if any(overflow for _, _, overflow in parts):
        return BoundedPrefixResult(
            stdout="",
            returncode=0,
            overflow=True,
            duration_s=time.monotonic() - started,
        )

    stdout_bytes = b"".join(output for output, _, _ in parts)
    if len(stdout_bytes) > max_bytes:
        return BoundedPrefixResult(
            stdout="",
            returncode=0,
            overflow=True,
            duration_s=time.monotonic() - started,
        )
    return BoundedPrefixResult(
        stdout=stdout_bytes.decode("utf-8"),
        returncode=0 if any(rc == 0 for _, rc, _ in parts) else parts[0][1],
        overflow=False,
        duration_s=time.monotonic() - started,
    )


def run_stages_on_text(
    stages: Iterable[Stage],
    stdin: str,
    *,
    timeout: float,
    env: dict | None,
) -> tuple[str, int]:
    """Run downstream pipeline stages against an in-memory prefix result."""
    command = _stage_text(stages)
    try:
        proc = run_pipeline(
            command,
            timeout=timeout,
            input=stdin,
            env=env,
        )
        stdout = proc.stdout or ""
        if proc.truncated:
            stdout += OUTPUT_TRUNCATION_MARKER
        return stdout, proc.returncode
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]", -1


def _merge_stderrs(parts: list[tuple[str, str, int]]) -> str:
    """Concatenate non-empty stderrs, deduplicated, so a common rg error
    (e.g. "regex parse error") shows once rather than N times."""
    seen = set()
    out_lines: list[str] = []
    for _, stderr, _ in parts:
        for line in stderr.splitlines():
            if line and line not in seen:
                seen.add(line)
                out_lines.append(line)
    return ("\n".join(out_lines) + "\n") if out_lines else ""


def _overall_rc(parts: list[tuple[str, str, int]]) -> int:
    """Returncode: 0 if any shard succeeded (matches rg single-file semantics
    where exit 1 means "no matches" and exit 0 means "matches found" — even
    if some shards have no matches, we want exit 0 overall when at least one
    found something). For total failure (all shards rc != 0), pick the most
    common non-zero code so eg. timeouts vs syntax errors are distinguishable.
    """
    if any(rc == 0 for _, _, rc in parts):
        return 0
    # All non-zero: report the first non-zero (typically all the same).
    for _, _, rc in parts:
        if rc != 0:
            return rc
    return 0  # empty input shouldn't happen but be safe


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def run_concat(plan: Plan, *, shard_files: list[str], corpus_token: str,
               timeout: float, env: dict | None) -> tuple[str, int]:
    """Stateless pipeline; concat per-shard outputs in shard order."""
    pipeline_text = _stage_text(plan.stages)
    cmds = [_sub_corpus_token(pipeline_text, corpus_token, sf) for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)
    stdout = "".join(p[0] for p in parts)
    return stdout, _overall_rc(parts)


def run_head(plan: Plan, *, shard_files: list[str], corpus_token: str,
             timeout: float, env: dict | None) -> tuple[str, int]:
    """CONCAT pipeline ending in `head -n N`. Apply `head -n N` per shard
    (bounds memory), concat, then trim again to N lines total."""
    assert plan.head_n is not None
    n = plan.head_n

    # Per-shard pipeline already ends in `head -n N`, so we keep the original
    # stages as-is. The per-shard rg→...→head pipeline already bounds output
    # to N lines per shard. After concat we'll have at most N * len(shards).
    pipeline_text = _stage_text(plan.stages)
    cmds = [_sub_corpus_token(pipeline_text, corpus_token, sf) for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)

    # Concat in shard order, then re-head to N (lines, not bytes — matches
    # `head -n N` semantics).
    concat = "".join(p[0] for p in parts)
    if not concat:
        return concat, _overall_rc(parts)

    lines = concat.split("\n")
    # `head -n N` returns the first N "lines" including the trailing newline
    # behavior. To match byte-equivalence with a single-file run we slice
    # by lines and rebuild. A trailing newline-only element from split() is
    # preserved to keep the original file's terminator.
    if len(lines) > n:
        # Take first N lines. Need to add the trailing newline if the original
        # nth line was newline-terminated (which it always is after split('\n')
        # except possibly the last element).
        head_lines = lines[:n]
        out = "\n".join(head_lines) + "\n"
    else:
        out = concat
    return out, _overall_rc(parts)


def run_count(plan: Plan, *, shard_files: list[str], corpus_token: str,
              timeout: float, env: dict | None) -> tuple[str, int]:
    """CONCAT pipeline ending in `wc -l`. Sum the per-shard counts.

    Each per-shard call returns "<n>\n" (the body of `wc -l` output when
    reading from stdin — no filename prefix).
    """
    pipeline_text = _stage_text(plan.stages)
    cmds = [_sub_corpus_token(pipeline_text, corpus_token, sf) for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)

    total = 0
    for stdout, _stderr, rc in parts:
        if rc != 0 and not stdout.strip():
            continue  # treat empty failed shard as 0 (matches `rg | wc -l`
                      # when rg exits 1 because no matches)
        try:
            total += int(stdout.strip().split()[0])
        except (ValueError, IndexError):
            # If a shard's wc output is malformed, surface as failure.
            return "", 1
    # `wc -l` from a single file with no filename prefix outputs e.g. "      42\n".
    # When stdin is piped, no filename and a single line: e.g. "42\n". Match the
    # piped form (which is what the model's pipelines always produce).
    return f"{total}\n", 0


def run_sort_head(plan: Plan, *, shard_files: list[str], corpus_token: str,
                  timeout: float, env: dict | None) -> tuple[str, int]:
    """CONCAT-body ending in `sort [| uniq] | head -n N`.

    Per shard: run [body] | sort | head -n N (uniq skipped per-shard — it'd
    only dedupe within a shard, must dedupe globally).
    Then: merge the N sorted streams with `sort -m`, optionally `uniq`, and
    `head -n N`.
    """
    assert plan.head_n is not None
    n = plan.head_n

    # Split the stages: everything up to (but not including) the trailing
    # `sort` is the per-shard body; we re-assemble the per-shard pipeline as
    # `<body> | sort | head -n N`.
    stages = list(plan.stages)
    tail_count = 3 if plan.has_uniq else 2  # sort | (uniq |) head
    body = stages[:-tail_count]
    if not body:
        # Pathological case (`sort | uniq | head -n 3` with no upstream
        # filter) — would need to read the whole corpus into sort. Bail.
        return run_concat(plan, shard_files=shard_files, corpus_token=corpus_token,
                          timeout=timeout, env=env)

    per_shard_pipeline_template = _stage_text(body) + f" | sort | head -n {n}"
    cmds = [_sub_corpus_token(per_shard_pipeline_template, corpus_token, sf)
            for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)

    # Write each shard's partial-sorted output to a tempfile so `sort -m`
    # can merge them. Could also pipe via process-substitution but tempfiles
    # are simpler and tolerate large outputs without blocking writers.
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="parallel_search_sortmerge_")
    try:
        files: list[str] = []
        for i, (stdout, _stderr, rc) in enumerate(parts):
            fp = os.path.join(tmpdir, f"part_{i:04d}")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(stdout)
            files.append(fp)
        # Build the merge pipeline. Quote each file path safely.
        import shlex as _shlex
        files_q = " ".join(_shlex.quote(p) for p in files)
        merge_cmd = f"sort -m {files_q}"
        if plan.has_uniq:
            merge_cmd += " | uniq"
        merge_cmd += f" | head -n {n}"
        try:
            mproc = run_pipeline(
                merge_cmd, timeout=timeout, env=env,
            )
            stdout = mproc.stdout or ""
            if mproc.truncated:
                stdout += OUTPUT_TRUNCATION_MARKER
            return stdout, mproc.returncode
        except subprocess.TimeoutExpired:
            return "", -1
    finally:
        # Clean up tempfiles.
        for f in files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


# Dispatch table.
STRATEGY_DISPATCH = {
    Strategy.CONCAT:    run_concat,
    Strategy.HEAD:      run_head,
    Strategy.COUNT:     run_count,
    Strategy.SORT_HEAD: run_sort_head,
    # FALLBACK is handled by the engine itself (runs against the single file).
}
