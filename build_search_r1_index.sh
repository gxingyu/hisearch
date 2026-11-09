#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_R1_ROOT="${SEARCH_R1_ROOT:-${REPO_ROOT}/../Search-R1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE=1
exec "$REPO_ROOT/venv/bin/python" "$SEARCH_R1_ROOT/search_r1/search/index_builder.py" \
  --retrieval_method e5 \
  --model_path "$REPO_ROOT/data/search_r1/e5-base-v2" \
  --corpus_path "$REPO_ROOT/data/wiki_18_corpus/wiki_corpus.jsonl" \
  --save_dir "$REPO_ROOT/data/search_r1/rebuilt_index" \
  --use_fp16 --max_length 256 --batch_size 512 --pooling_method mean --faiss_type Flat "$@"
