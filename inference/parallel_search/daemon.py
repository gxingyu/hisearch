# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
"""Long-running search-engine daemon.

Holds one ShardedSearchEngine in-process; accepts shell-command requests over
a Unix domain socket; returns results in the same shape the in-proc engine
produces. It coalesces identical concurrent requests, caches complete results,
and shares bounded first-stage scans across multi-filter pipelines. A
configurable query-level concurrency cap prevents cache misses from multiplying
into an unbounded number of rg processes.

Broad first-stage outputs automatically fall back to the original sharded
pipeline. This keeps memory bounded while preserving byte-equivalent results.

Protocol: length-prefixed JSON, both directions.
    <8-byte big-endian uint64 length><utf-8 JSON body>

Request body:
    {"command": str, "env_overrides": dict | None}
Response body:
    {"stdout": str, "returncode": int, "strategy": str,
     "fallback_reason": str, "cache_status": str,
     "first_stage_cache_status": str}

Launch standalone:
    python -m inference.parallel_search.daemon \\
        --socket /tmp/search.sock \\
        --shard_dir /scratch/.../shards_16 \\
        --corpus /scratch/.../wiki_corpus.jsonl

Or auto-spawned by scripts/rl/eval_rl_fast.sh.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import socketserver
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .cache import SingleFlightLRU
from .engine import EngineStats, ShardedSearchEngine
from .pipeline import Plan, Strategy
from .runners import BoundedPrefixResult, run_first_stage_bounded, run_stages_on_text
from .subprocesses import bound_output

logger = logging.getLogger(__name__)

# Wire-protocol: 8-byte unsigned big-endian length prefix + utf-8 JSON body.
_LEN_FMT = "!Q"
_LEN_SIZE = struct.calcsize(_LEN_FMT)


# ---------------------------------------------------------------------------
# Wire helpers (used by both daemon and client)
# ---------------------------------------------------------------------------

def send_msg(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(_LEN_FMT, len(body)))
    sock.sendall(body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    """Read one length-prefixed JSON message. Returns None on clean EOF."""
    try:
        header = sock.recv(_LEN_SIZE, socket.MSG_WAITALL)
    except (ConnectionError, OSError):
        return None
    if len(header) == 0:
        return None
    if len(header) < _LEN_SIZE:
        # Partial header → finish with the exact-read helper.
        header = bytes(header) + _recv_exact(sock, _LEN_SIZE - len(header))
    (length,) = struct.unpack(_LEN_FMT, header)
    body = _recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchValue:
    stdout: str
    returncode: int
    strategy: str
    fallback_reason: str = ""
    first_stage_cache_status: str = "not_eligible"


class SearchCoordinator:
    """Two-level cache and single-flight coordinator for daemon queries."""

    def __init__(
        self,
        engine: ShardedSearchEngine,
        query_slots: threading.BoundedSemaphore,
        *,
        result_cache_bytes: int,
        first_stage_cache_bytes: int,
        first_stage_max_entry_bytes: int,
    ) -> None:
        self.engine = engine
        self.query_slots = query_slots
        self.first_stage_max_entry_bytes = first_stage_max_entry_bytes
        self.result_cache = SingleFlightLRU[_SearchValue](
            result_cache_bytes,
            lambda value: len(value.stdout.encode("utf-8")) + 256,
        )
        self.first_stage_cache = SingleFlightLRU[BoundedPrefixResult](
            first_stage_cache_bytes,
            lambda value: 1 if value.overflow else len(value.stdout.encode("utf-8")) + 256,
        )
        self._request_count = 0
        self._count_lock = threading.Lock()

    @staticmethod
    def _environment_key(env_overrides: dict | None) -> tuple[tuple[str, str], ...]:
        if not env_overrides:
            return ()
        return tuple(sorted((str(key), str(value)) for key, value in env_overrides.items()))

    @staticmethod
    def _normalize_first_stage(stage_text: str, corpus_token: str) -> str:
        return stage_text.replace(corpus_token, "{CORPUS}")

    def execute(
        self,
        command: str,
        env: dict | None,
        env_overrides: dict | None,
    ) -> tuple[_SearchValue, EngineStats]:
        started = time.monotonic()
        environment_key = self._environment_key(env_overrides)
        cache_key = (command, environment_key)

        value, cache_status = self.result_cache.get_or_compute(
            cache_key,
            lambda: self._execute_uncached(command, env, environment_key),
        )
        first_stage_status = (
            value.first_stage_cache_status if cache_status == "miss" else "not_checked"
        )
        stats = EngineStats(
            strategy=value.strategy,
            n_shards_used=self.engine.n_shards,
            fallback_reason=value.fallback_reason,
            duration_s=time.monotonic() - started,
            cache_status=cache_status,
            first_stage_cache_status=first_stage_status,
        )
        self.engine.record_query(command, value.returncode, stats)
        self._maybe_log_cache_stats()
        return value, stats

    def _execute_uncached(
        self,
        command: str,
        env: dict | None,
        environment_key: tuple[tuple[str, str], ...],
    ) -> tuple[_SearchValue, bool]:
        corpus_token, raw_plan = self.engine.classify_command(command)
        plan = raw_plan if isinstance(raw_plan, Plan) else None
        if (
            corpus_token is not None
            and plan is not None
            and plan.strategy == Strategy.HEAD
            and len(plan.stages) >= 3
            and self.first_stage_max_entry_bytes > 0
        ):
            first_stage_key = (
                self._normalize_first_stage(plan.stages[0].raw, corpus_token),
                environment_key,
            )
            prefix, first_stage_status = self.first_stage_cache.get_or_compute(
                first_stage_key,
                lambda: self._scan_first_stage(plan, corpus_token, env),
            )
            if not prefix.overflow and prefix.returncode != -1:
                stdout, returncode = run_stages_on_text(
                    plan.stages[1:],
                    prefix.stdout,
                    timeout=self.engine.timeout,
                    env=env,
                )
                value = _SearchValue(
                    stdout=bound_output(stdout),
                    returncode=returncode,
                    strategy=plan.strategy.value,
                    first_stage_cache_status=first_stage_status,
                )
                return value, returncode != -1
            first_stage_status = (
                f"{first_stage_status}_too_large"
                if prefix.overflow
                else f"{first_stage_status}_failed"
            )
        else:
            first_stage_status = "not_eligible"

        with self.query_slots:
            stdout, returncode = self.engine.execute(command, env=env, record=False)
            engine_stats = self.engine.last_stats
        value = _SearchValue(
            stdout=bound_output(stdout),
            returncode=returncode,
            strategy=engine_stats.strategy,
            fallback_reason=engine_stats.fallback_reason,
            first_stage_cache_status=first_stage_status,
        )
        return value, returncode != -1

    def _scan_first_stage(
        self,
        plan: Plan,
        corpus_token: str,
        env: dict | None,
    ) -> tuple[BoundedPrefixResult, bool]:
        with self.query_slots:
            result = run_first_stage_bounded(
                plan.stages[0],
                shard_files=self.engine.shard_files,
                corpus_token=corpus_token,
                timeout=self.engine.timeout,
                env=env,
                max_bytes=self.first_stage_max_entry_bytes,
            )
        cacheable = result.overflow or result.returncode != -1
        return result, cacheable

    def _maybe_log_cache_stats(self) -> None:
        with self._count_lock:
            self._request_count += 1
            request_count = self._request_count
        if request_count % 1000 != 0:
            return
        logger.info(
            "cache stats after %d requests: result=%s first_stage=%s",
            request_count,
            self.result_cache.info(),
            self.first_stage_cache.info(),
        )


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        # Keep the connection open for many requests (typical eval worker
        # reuses the same socket). A dropped peer closes the loop cleanly.
        while True:
            try:
                msg = recv_msg(sock)
            except (json.JSONDecodeError, ConnectionError, OSError) as exc:
                logger.debug(f"daemon: recv failed: {exc}")
                return
            if msg is None:
                return
            cmd = msg.get("command")
            env_overrides = msg.get("env_overrides")  # dict or None
            if not isinstance(cmd, str):
                self._reply_error(sock, "missing or non-string 'command' field")
                continue
            env = None
            if env_overrides:
                env = {**os.environ, **env_overrides}
            try:
                value, stats = self.server.search.execute(cmd, env, env_overrides)
                send_msg(sock, {
                    "stdout": value.stdout,
                    "returncode": value.returncode,
                    "strategy": stats.strategy,
                    "fallback_reason": stats.fallback_reason,
                    "duration_s": stats.duration_s,
                    "cache_status": stats.cache_status,
                    "first_stage_cache_status": stats.first_stage_cache_status,
                })
            except Exception as exc:
                logger.exception("daemon: engine.execute raised")
                self._reply_error(sock, f"engine error: {type(exc).__name__}: {exc}")

    @staticmethod
    def _reply_error(sock: socket.socket, reason: str) -> None:
        try:
            send_msg(sock, {
                "stdout": "",
                "returncode": -2,
                "strategy": "error",
                "fallback_reason": reason,
            })
        except (OSError, BrokenPipeError):
            pass


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256

    def __init__(self, server_address: str, RequestHandlerClass, engine,
                 max_inflight_queries: int = 1,
                 result_cache_bytes: int = 512 << 20,
                 first_stage_cache_bytes: int = 2 << 30,
                 first_stage_max_entry_bytes: int = 64 << 20):
        super().__init__(server_address, RequestHandlerClass)
        self.engine = engine
        if max_inflight_queries < 1:
            raise ValueError("max_inflight_queries must be >= 1")
        self.query_slots = threading.BoundedSemaphore(max_inflight_queries)
        self.search = SearchCoordinator(
            engine,
            self.query_slots,
            result_cache_bytes=result_cache_bytes,
            first_stage_cache_bytes=first_stage_cache_bytes,
            first_stage_max_entry_bytes=first_stage_max_entry_bytes,
        )


def serve(socket_path: str, engine: ShardedSearchEngine,
          pid_file: Optional[str] = None,
          max_inflight_queries: int = 1,
          result_cache_bytes: int = 512 << 20,
          first_stage_cache_bytes: int = 2 << 30,
          first_stage_max_entry_bytes: int = 64 << 20) -> _ThreadedUnixServer:
    """Start serving on `socket_path`. Blocks until SIGTERM/SIGINT.

    Returns the server object (for tests that want to call .shutdown() from
    another thread). In production, this function does not return — the
    signal handlers call .shutdown() and then we clean up.
    """
    # If a stale socket file exists (previous crashed run), remove it. Don't
    # touch a socket that's actively in use — bind would fail and we'd notice.
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    server = _ThreadedUnixServer(
        socket_path,
        _Handler,
        engine,
        max_inflight_queries=max_inflight_queries,
        result_cache_bytes=result_cache_bytes,
        first_stage_cache_bytes=first_stage_cache_bytes,
        first_stage_max_entry_bytes=first_stage_max_entry_bytes,
    )
    # Lock down the socket so other cluster users can't connect.
    try:
        os.chmod(socket_path, 0o600)
    except OSError:
        pass

    if pid_file:
        try:
            with open(pid_file, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
        except OSError as exc:
            logger.warning(f"daemon: failed to write pid_file {pid_file}: {exc}")

    def _shutdown(signum, _frame) -> None:
        logger.info(f"daemon: signal {signum} received, shutting down")
        # serve_forever's shutdown must run on a non-server thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"daemon: serving on {socket_path} ({engine.describe()})")
    try:
        server.serve_forever()
    finally:
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        if pid_file:
            try:
                os.unlink(pid_file)
            except OSError:
                pass
        logger.info("daemon: stopped")
    return server


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", required=True, help="Unix socket path to bind")
    p.add_argument("--shard_dir", required=True,
                   help="directory containing pre-built shards + manifest.json")
    p.add_argument("--corpus", required=True,
                   help="absolute path to fallback corpus file (e.g. wiki_corpus.jsonl)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-call subprocess timeout (seconds)")
    p.add_argument("--engine_log", default=None,
                   help="optional jsonl path for per-call engine telemetry")
    p.add_argument("--pid_file", default=None,
                   help="optional path to write the daemon's PID")
    p.add_argument("--max-inflight-queries", type=int, default=1,
                   help="maximum concurrent whole-query shard fan-outs (default: 1)")
    p.add_argument("--result-cache-mb", type=int, default=512,
                   help="LRU capacity for complete command results (default: 512 MiB)")
    p.add_argument("--first-stage-cache-mb", type=int, default=2048,
                   help="LRU capacity for shared first-stage results (default: 2048 MiB)")
    p.add_argument("--first-stage-max-entry-mb", type=int, default=64,
                   help="maximum cached output for one first-stage scan (default: 64 MiB)")
    p.add_argument("--warmup", action="store_true",
                   help="read all shards once before accepting queries")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = ShardedSearchEngine(
        shard_dir=args.shard_dir,
        fallback_corpus_file=args.corpus,
        timeout=args.timeout,
        log_path=args.engine_log,
    )
    if args.warmup:
        warmed = engine.warm_cache()
        logger.info("warmed %d bytes across %d shards", warmed, engine.n_shards)
    serve(
        args.socket,
        engine,
        pid_file=args.pid_file,
        max_inflight_queries=args.max_inflight_queries,
        result_cache_bytes=max(0, args.result_cache_mb) << 20,
        first_stage_cache_bytes=max(0, args.first_stage_cache_mb) << 20,
        first_stage_max_entry_bytes=max(0, args.first_stage_max_entry_mb) << 20,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
