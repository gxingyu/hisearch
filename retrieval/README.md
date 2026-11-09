# Frozen dense retrieval service

The service reuses the Search-R1 E5 encoder and serves FAISS results through
`POST /retrieve`. The policy tool always requests three passages. The HTTP
service additionally accepts batched queries and configurable top-k for
diagnostics; this does not expose top-k control to the policy.

## Separate environment and resources

Use a separate Python 3.10 retrieval environment. The reference configuration
uses PyTorch 2.4.0/CUDA 12.1, FAISS GPU 1.8.0, and transformers 4.48.3; it is
separate from the policy-training environment.

```bash
# From the HiSearch repository root. Install uv separately, or create this
# environment manually with the versions in retrieval/requirements.txt.
git clone https://github.com/PeterGriffinJin/Search-R1.git ../Search-R1

# Download the raw corpus first. This downloader handles the published archive.
python sft/data_generation/download_corpus.py --dest data/wiki_18_corpus
bash setup_search_r1.sh
```

`setup_search_r1.sh` creates `venv/`, installs the retrieval dependencies, and
downloads/prepares the frozen E5 model, index, corpus offsets, and manifest.
It does not launch the service. It uses `uv`; the corpus downloader needs
`huggingface_hub` in the calling Python environment.

Resources are placed under:

```text
data/wiki_18_corpus/wiki_corpus.jsonl
data/search_r1/e5-base-v2/
data/search_r1/index/e5_Flat.index
data/search_r1/corpus_offsets.npy
data/search_r1/corpus_manifest.json
```

The public sources are `PeterJinGo/wiki-18-corpus`, `intfloat/e5-base-v2`,
and `PeterJinGo/wiki-18-e5-index`. The included downloader uses the
[ModelScope index mirror](https://www.modelscope.cn/datasets/yamseyoung/wiki-18-e5-index)
and verifies each part against the pinned SHA256 values in `prepare.py` before
merging. Model, index, and corpus source revisions are recorded in code.
Use `HISEARCH_INDEX_PROXY` or `--index-proxy` when a proxy is needed; there is
no built-in private proxy address.

The corpus is about 14 GB and the merged index about 64.6 GB. Keeping both index
parts and the merged file requires about 129 GB for the index alone, plus models,
corpus, environments, and temporary space. GPU FAISS shards use FP16 copies;
allocate GPU and host memory according to your hardware.

## Start and inspect

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_search_r1.sh \
  --search-r1-root ../Search-R1 --batch-size 32 --batch-wait-ms 10
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Who is Marie Curie married to?"],"topk":3,"return_scores":true}'
```

The endpoint returns Search-R1-style documents and scores. `/health` exposes
the corpus checksum/count, batch settings, and parallel-shard mode. Startup
checks passage/index alignment at sampled locations; this is not an exhaustive
content verification of every vector.

The hybrid launcher requires matching corpus SHA256/row counts, fixed top-3
retrieval, and the expected batching/parallel settings. If training and dense
retrieval share GPUs, reserve memory for the frozen index and encoder as well
as the policy. Separate GPUs or a remote compatible endpoint can also be used.

Rebuilding the index is optional and expensive:

```bash
bash build_search_r1_index.sh
bash run_search_r1.sh --index data/search_r1/rebuilt_index/e5_Flat.index
```

`--cpu-index` keeps FAISS on CPU; the query encoder still needs CUDA in this
implementation. Downloading, full-corpus indexing, and GPU serving were not
rerun as part of source-release validation.
