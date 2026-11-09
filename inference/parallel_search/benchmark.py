from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cache import SingleFlightLRU
from .daemon import SearchCoordinator
from .engine import ShardedSearchEngine
from .pipeline import Plan, Strategy


def replay(engine, commands, concurrency, cache_mb, clients, profile_capacities=()):
    coordinator = SearchCoordinator(
        engine,
        threading.BoundedSemaphore(concurrency),
        result_cache_bytes=512 << 20,
        first_stage_cache_bytes=cache_mb << 20,
        first_stage_max_entry_bytes=64 << 20,
    )
    prefix_weights = {}
    result_weights = {}
    scan_first_stage = coordinator._scan_first_stage
    completion_lock = threading.Lock()
    completed_queries = 0

    def profiled_scan(plan, corpus_token, env):
        result, cacheable = scan_first_stage(plan, corpus_token, env)
        key = coordinator._normalize_first_stage(plan.stages[0].raw, corpus_token)
        prefix_weights[key] = 1 if result.overflow else len(result.stdout.encode()) + 256
        return result, cacheable

    coordinator._scan_first_stage = profiled_scan

    def query(command):
        nonlocal completed_queries
        started = time.monotonic()
        value, _ = coordinator.execute(command, None, None)
        result_weights[command] = len(value.stdout.encode()) + 256
        digest = hashlib.sha256(value.stdout.encode()).hexdigest()
        with completion_lock:
            completed_queries += 1
            if completed_queries % 500 == 0:
                print(f"Completed {completed_queries}/{len(commands)} queries", flush=True)
        return time.monotonic() - started, digest, value.returncode

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=clients) as executor:
        results = list(executor.map(query, commands))
    wall = time.monotonic() - started
    durations = sorted(result[0] for result in results)
    summary = {
        "queries": len(commands),
        "clients": clients,
        "concurrency": concurrency,
        "first_stage_cache_mb": cache_mb,
        "wall_seconds": wall,
        "queries_per_second": len(commands) / wall,
        "mean_seconds": statistics.mean(durations),
        "p50_seconds": statistics.median(durations),
        "p95_seconds": durations[int((len(durations) - 1) * 0.95)],
        "p99_seconds": durations[int((len(durations) - 1) * 0.99)],
        "result_cache": coordinator.result_cache.info(),
        "first_stage_cache": coordinator.first_stage_cache.info(),
        "failed_queries": sum(result[2] not in (0, 1) for result in results),
    }
    if profile_capacities:
        profiles = []
        for capacity in profile_capacities:
            complete_cache = SingleFlightLRU[int](512 << 20, lambda weight: weight)
            prefix_cache = SingleFlightLRU[int](capacity << 20, lambda weight: weight)
            for command in commands:
                _, status = complete_cache.get_or_compute(command, lambda: (result_weights[command], True))
                if status == "hit":
                    continue
                corpus_token, plan = engine.classify_command(command)
                if corpus_token is None or not isinstance(plan, Plan) or plan.strategy != Strategy.HEAD or len(plan.stages) < 3:
                    continue
                key = coordinator._normalize_first_stage(plan.stages[0].raw, corpus_token)
                if key in prefix_weights:
                    prefix_cache.get_or_compute(key, lambda: (prefix_weights[key], True))
            profiles.append({"cache_mb": capacity, **prefix_cache.info()})
        summary["capacity_simulation"] = profiles
        summary["measured_unique_prefix_bytes"] = sum(prefix_weights.values())
    print(json.dumps(summary), flush=True)
    return summary, [(result[1], result[2]) for result in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--clients", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--cache-mb", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--profile-capacities", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.queries < 1 or args.clients < 1:
        parser.error("queries and clients must be positive")
    os.environ["GREPSEEK_TOOL_MAX_OUTPUT_BYTES"] = "65536"
    with args.log.open() as stream:
        commands = [json.loads(line)["command"] for _, line in zip(range(args.queries), stream)]
    engine = ShardedSearchEngine(shard_dir=args.shard_dir, fallback_corpus_file=args.corpus, timeout=180)
    print("Warming corpus page cache...", flush=True)
    engine.warm_cache()
    summaries = []
    reference = None
    for concurrency in args.concurrency:
        for cache_mb in args.cache_mb:
            print(f"Replaying {len(commands)} queries: concurrency={concurrency}, cache={cache_mb} MiB", flush=True)
            summary, results = replay(engine, commands, concurrency, cache_mb, args.clients, args.profile_capacities)
            if reference is None:
                reference = results
            summary["result_mismatches"] = sum(actual != expected for actual, expected in zip(results, reference))
            summaries.append(summary)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
