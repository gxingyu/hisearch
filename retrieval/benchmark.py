import argparse
import json
import math
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


def request_json(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="Replay real dense queries with bounded HTTP concurrency")
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.num_queries < 1 or args.concurrency < 1:
        parser.error("num-queries and concurrency must be positive")
    queries = []
    with args.rollouts.open() as stream:
        for line in stream:
            row = json.loads(line)
            for body in re.findall(r"<tool_call>(.*?)</tool_call>", row.get("output", ""), re.S):
                try:
                    call = json.loads(body)
                except ValueError:
                    continue
                if call.get("name") == "dense_search":
                    query = call.get("arguments", {}).get("query")
                    if isinstance(query, str) and query.strip():
                        queries.append(query)
    random.Random(42).shuffle(queries)
    queries = queries[: args.num_queries]
    if len(queries) < args.num_queries:
        parser.error("rollouts do not contain enough dense queries")
    parsed = urlsplit(args.url)
    health = request_json(urlunsplit((parsed.scheme, parsed.netloc, "/health", "", "")), timeout=args.timeout)
    request_json(args.url, {"queries": [queries[0]], "topk": 3, "return_scores": True}, args.timeout)

    def retrieve(query):
        started = time.perf_counter()
        response = request_json(args.url, {"queries": [query], "topk": 3, "return_scores": True}, args.timeout)
        elapsed = time.perf_counter() - started
        batches = response.get("result")
        if not isinstance(batches, list) or len(batches) != 1 or len(batches[0]) != 3:
            raise ValueError("Expected exactly three passages per query")
        documents = batches[0]
        if not all(math.isfinite(item["score"]) and isinstance(item["document"]["contents"], str) for item in documents):
            raise ValueError("Invalid retrieval result")
        return {"query": query, "seconds": elapsed,
                "passage_ids": [str(item["document"]["id"]) for item in documents],
                "scores": [item["score"] for item in documents]}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        samples = list(executor.map(retrieve, queries))
    elapsed = time.perf_counter() - started
    durations = sorted(sample["seconds"] for sample in samples)
    summary = {"queries": len(samples), "concurrency": args.concurrency, "topk": 3,
               "mean_seconds": statistics.mean(durations), "median_seconds": statistics.median(durations),
               "p95_seconds": durations[math.ceil(len(durations) * .95) - 1],
               "max_seconds": durations[-1], "wall_seconds": elapsed, "queries_per_second": len(samples) / elapsed,
               "health": health}
    if args.reference:
        reference = json.loads(args.reference.read_text())["samples"]
        if [sample["query"] for sample in reference] != queries:
            raise ValueError("Reference uses different queries")
        summary["exact_top3_matches"] = sum(
            sample["passage_ids"] == previous["passage_ids"] for sample, previous in zip(samples, reference)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
