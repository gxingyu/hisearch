import argparse
import asyncio
import importlib.util
import json
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Full

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from corpus import Corpus
from batching import RetrievalBatcher


class QueryRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=128)
    topk: int | None = Field(default=None, ge=1, le=50)
    return_scores: bool = False


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--search-r1-root", type=Path, default=root.parent / "Search-R1")
    parser.add_argument("--data-root", type=Path, default=root / "data/search_r1")
    parser.add_argument("--corpus", type=Path, default=root / "data/wiki_18_corpus/wiki_corpus.jsonl")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu-index", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--serial-shards", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_wait_ms < 0 or args.queue_size < 1:
        parser.error("batch-size and queue-size must be positive; batch-wait-ms must be nonnegative")
    module_path = args.search_r1_root / "search_r1/search/retrieval_server.py"
    specification = importlib.util.spec_from_file_location("search_r1_retrieval", module_path)
    upstream = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(upstream)
    manifest = json.loads((args.data_root / "corpus_manifest.json").read_text())
    stat = args.corpus.stat()
    if (stat.st_size, stat.st_mtime_ns) != (manifest["size"], manifest["mtime_ns"]):
        raise ValueError("Corpus changed; rerun retrieval/prepare.py")
    corpus = Corpus(args.corpus, args.data_root / "corpus_offsets.npy")
    index = faiss.read_index(str(args.index or args.data_root / "index/e5_Flat.index"))
    if index.ntotal != len(corpus) or index.d != 768 or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Index/corpus mismatch")
    encoder = upstream.Encoder("e5", str(args.data_root / "e5-base-v2"), "mean", 256, True)
    positions = [0, 29, 100, len(corpus) // 4, len(corpus) // 2, len(corpus) - 1]
    embeddings = encoder.encode([corpus[position]["contents"] for position in positions], is_query=False)
    vectors = np.stack([index.reconstruct(position) for position in positions])
    similarities = (embeddings * vectors).sum(axis=1) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vectors, axis=1))
    if not np.all(similarities > 0.99):
        raise ValueError(f"Corpus/index sample alignment failed: {similarities.tolist()}")
    validation = {"rows": len(corpus), "dimension": index.d, "corpus_sha256": manifest["sha256"],
                  "sample_positions": positions, "sample_cosines": similarities.tolist()}
    (args.data_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation), flush=True)
    if args.validate_only:
        corpus.close()
        return
    if not args.cpu_index:
        gpu_count = faiss.get_num_gpus()
        if gpu_count < 1:
            raise ValueError("No visible CUDA GPUs")
        resources = []
        shards = []
        cpu_vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
        for device in range(gpu_count):
            resource = faiss.StandardGpuResources()
            resource.setTempMemory(256 * 1024 * 1024)
            configuration = faiss.GpuIndexFlatConfig()
            configuration.device = device
            configuration.useFloat16 = True
            shard = faiss.GpuIndexFlatIP(resource, index.d, configuration)
            start = index.ntotal * device // gpu_count
            end = index.ntotal * (device + 1) // gpu_count
            shard.add(cpu_vectors[start:end])
            resources.append(resource)
            shards.append(shard)
        gpu_index = faiss.IndexShards(index.d, not args.serial_shards, True)
        for shard in shards:
            gpu_index.add_shard(shard)
        del cpu_vectors
        index = gpu_index
    def retrieve_batch(queries, topk):
        embeddings = encoder.encode(queries)
        scores, positions = index.search(embeddings, topk)
        results = []
        for row_positions, row_scores in zip(positions, scores):
            documents = []
            for position, score in zip(row_positions, row_scores):
                if position >= 0:
                    documents.append({"document": corpus[position], "score": float(score)})
            results.append(documents)
        return results

    batcher = RetrievalBatcher(retrieve_batch, args.batch_size, args.batch_wait_ms, args.queue_size)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.to_thread(batcher.close)
            corpus.close()

    app = FastAPI(title="Search-R1 dense retrieval", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "rows": len(corpus), "dimension": 768, "model": "intfloat/e5-base-v2",
                "corpus_sha256": manifest["sha256"], "gpu_index": not args.cpu_index,
                "batch_size": args.batch_size, "batch_wait_ms": args.batch_wait_ms,
                "parallel_shards": not args.cpu_index and not args.serial_shards}

    @app.post("/retrieve")
    async def retrieve(request: QueryRequest):
        pending = []
        try:
            for query in request.queries:
                pending.append(batcher.submit(query, request.topk or 3))
        except (Full, RuntimeError) as exc:
            for future in pending:
                future.cancel()
            raise HTTPException(status_code=503, detail="Retrieval queue is unavailable") from exc
        results = await asyncio.gather(*(asyncio.wrap_future(future) for future in pending))
        if not request.return_scores:
            results = [[item["document"] for item in documents] for documents in results]
        return {"result": results}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
