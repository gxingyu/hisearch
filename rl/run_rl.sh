#!/usr/bin/env bash
# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
# GRPO RL training for GrepSeek via verl's `main_ppo` (verl is the engine; the
# agent loop, search/grep tool, reward, and dataset are GrepSeek's, registered
# through the configs under grepseek/trainer/config). Reads the config at
# grepseek/trainer/config/grpo_trainer.yaml — the paper's "ours" GRPO run.
#
# Quick start (4×A100-80GB):
#   export GREPSEEK_MODEL_PATH=/path/to/sft_checkpoint/global_step_NNN/huggingface
#   export GREPSEEK_TRAIN_FILES=/path/to/rl/train.jsonl
#   export GREPSEEK_VAL_FILES=/path/to/rl/val.jsonl  # defaults to the 7-benchmark mix
#   export GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus      # dir holding wiki_corpus.jsonl
#   export NPROC=4
#   bash rl/run_rl.sh
#
# All knobs are environment variables (see below). Extra args are passed through
# to main_ppo, e.g.:  bash rl/run_rl.sh actor_rollout_ref.actor.optim.lr=2e-6
# Runs on any machine with >=2 GPUs once your training env is active.
set -euo pipefail
REPO_ROOT="${PWD}"                   # run launchers from the repo root
if [[ ! -f "${REPO_ROOT}/README.md" || ! -d "${REPO_ROOT}/sft" || ! -d "${REPO_ROOT}/rl" || ! -d "${REPO_ROOT}/inference" ]]; then
  echo "error: run this command from the grepseek repo root, e.g.:" >&2
  echo "       bash rl/run_rl.sh" >&2
  exit 2
fi
root_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}
export GREPSEEK_ROOT="${REPO_ROOT}"

# --- verl + grepseek on PYTHONPATH (both used via PYTHONPATH, not pip-installed) ---
VERL_DIR="${VERL_DIR:-${REPO_ROOT}/verl}"
if [[ ! -d "${VERL_DIR}/verl/trainer" ]]; then
  echo "error: vendored verl not found at ${VERL_DIR} (expected ${VERL_DIR}/verl/trainer)" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/rl:${VERL_DIR}:${PYTHONPATH:-}"

# --- caches (keep off the home directory) ---
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
# Triton autotuner races across Ray/torchrun workers on shared filesystems; pin
# it to local scratch (override TRITON_CACHE_DIR for node-local disk).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$$}"
mkdir -p "${TRITON_CACHE_DIR}"
# CUDA allocator: do NOT set expandable_segments for RL. vLLM's CuMemAllocator
# (the hybrid-engine memory pool that swaps weights between FSDP-train and vLLM
# rollout) is incompatible with it and crashes the rollout engine with
# "Expandable segments are not compatible with memory pool" (pytorch #147851).
# Leave the allocator at its default. (The SFT launcher DOES set it — SFT has no
# vLLM in the loop — but RL must not.)
export TOKENIZERS_PARALLELISM=true

# --- required paths ---
: "${GREPSEEK_TRAIN_FILES:?set GREPSEEK_TRAIN_FILES to the RL train JSONL}"
: "${GREPSEEK_VAL_FILES:=data/rl/flashrag_7bench/val.jsonl}"
GREPSEEK_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT:-${REPO_ROOT}/data/wiki_18_corpus}"
GREPSEEK_TRAIN_FILES="$(root_path "${GREPSEEK_TRAIN_FILES}")"
GREPSEEK_VAL_FILES="$(root_path "${GREPSEEK_VAL_FILES}")"
GREPSEEK_CORPUS_ROOT="$(root_path "${GREPSEEK_CORPUS_ROOT}")"
export GREPSEEK_TRAIN_FILES GREPSEEK_VAL_FILES
# The search tool reads VERL_TOOL_CORPUS_ROOT directly (takes precedence over the
# yaml corpus_root). It expects ${GREPSEEK_CORPUS_ROOT}/wiki_corpus.jsonl to exist.
export VERL_TOOL_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT}"
if [[ ! -f "${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl" ]]; then
  echo "error: ${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl not found." >&2
  echo "       Download it with sft/data_generation/download_corpus.py (see README)." >&2
  exit 1
fi

GREPSEEK_SEARCH_ENGINE="${GREPSEEK_SEARCH_ENGINE:-daemon}"
GREPSEEK_SEARCH_SHARDS="${GREPSEEK_SEARCH_SHARDS:-32}"
GREPSEEK_SEARCH_SHARD_DIR="${GREPSEEK_SEARCH_SHARD_DIR:-${GREPSEEK_CORPUS_ROOT}/.grepseek_shards_${GREPSEEK_SEARCH_SHARDS}}"
GREPSEEK_SEARCH_SOCKET="${GREPSEEK_SEARCH_SOCKET:-${GREPSEEK_TMP_ROOT:-/tmp}/grepseek_search_${USER:-user}_$$.sock}"
GREPSEEK_SEARCH_MAX_INFLIGHT="${GREPSEEK_SEARCH_MAX_INFLIGHT:-4}"
GREPSEEK_SEARCH_TIMEOUT="${GREPSEEK_SEARCH_TIMEOUT:-180}"
GREPSEEK_SEARCH_WARMUP="${GREPSEEK_SEARCH_WARMUP:-1}"
GREPSEEK_SEARCH_RESULT_CACHE_MB="${GREPSEEK_SEARCH_RESULT_CACHE_MB:-512}"
GREPSEEK_SEARCH_FIRST_STAGE_CACHE_MB="${GREPSEEK_SEARCH_FIRST_STAGE_CACHE_MB:-2048}"
GREPSEEK_SEARCH_FIRST_STAGE_MAX_ENTRY_MB="${GREPSEEK_SEARCH_FIRST_STAGE_MAX_ENTRY_MB:-64}"
export GREPSEEK_TOOL_MAX_OUTPUT_BYTES="${GREPSEEK_TOOL_MAX_OUTPUT_BYTES:-65536}"
GREPSEEK_TEST_FREQ="${GREPSEEK_TEST_FREQ:-500}"
GREPSEEK_VAL_MAX_SAMPLES="${GREPSEEK_VAL_MAX_SAMPLES:--1}"
GREPSEEK_SEARCH_SHARD_DIR="$(root_path "${GREPSEEK_SEARCH_SHARD_DIR}")"
export GREPSEEK_SEARCH_ENGINE GREPSEEK_SEARCH_SHARDS GREPSEEK_SEARCH_SHARD_DIR
export GREPSEEK_SEARCH_SOCKET GREPSEEK_SEARCH_MAX_INFLIGHT GREPSEEK_SEARCH_TIMEOUT
export GREPSEEK_SEARCH_WARMUP

# --- model / output / GPUs ---
export GREPSEEK_MODEL_PATH="${GREPSEEK_MODEL_PATH:-Qwen/Qwen3.5-9B}"
[[ "${GREPSEEK_MODEL_PATH}" == "Qwen/Qwen3.5-9B" ]] && \
  echo "note: GREPSEEK_MODEL_PATH unset -> initializing RL from the BASE model. The paper inits from the cold-start SFT checkpoint."
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NPROC="${NPROC:-4}"
ULYSSES_SP="${ULYSSES_SP:-2}"   # must divide NPROC; paper used 2 on 4 GPUs
export GREPSEEK_OUTPUT_DIR="${GREPSEEK_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/rl/$(date +%Y%m%d-%H%M%S)}"
GREPSEEK_OUTPUT_DIR="$(root_path "${GREPSEEK_OUTPUT_DIR}")"
export GREPSEEK_OUTPUT_DIR
mkdir -p "${GREPSEEK_OUTPUT_DIR}"

SEARCH_DAEMON_PID=""
cleanup_search_daemon() {
  if [[ -n "${SEARCH_DAEMON_PID}" ]] && kill -0 "${SEARCH_DAEMON_PID}" 2>/dev/null; then
    kill "${SEARCH_DAEMON_PID}" 2>/dev/null || true
    wait "${SEARCH_DAEMON_PID}" 2>/dev/null || true
  fi
}
trap cleanup_search_daemon EXIT

if [[ "${GREPSEEK_SEARCH_ENGINE}" == "daemon" && "${GREPSEEK_SEARCH_EXTERNAL_DAEMON:-0}" == "1" ]]; then
  [[ -S "${GREPSEEK_SEARCH_SOCKET}" ]] || {
    echo "error: external search daemon socket is missing: ${GREPSEEK_SEARCH_SOCKET}" >&2
    exit 1
  }
elif [[ "${GREPSEEK_SEARCH_ENGINE}" == "daemon" ]]; then
  if [[ -e "${GREPSEEK_SEARCH_SOCKET}" ]]; then
    echo "error: search socket already exists: ${GREPSEEK_SEARCH_SOCKET}" >&2
    echo "       set GREPSEEK_SEARCH_SOCKET to a fresh path or remove the stale socket." >&2
    exit 1
  fi
  DAEMON_WARMUP_ARGS=()
  [[ "${GREPSEEK_SEARCH_WARMUP}" == "1" ]] && DAEMON_WARMUP_ARGS+=(--warmup)
  python -m inference.parallel_search.sharder \
      --src "${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl" \
      --dst "${GREPSEEK_SEARCH_SHARD_DIR}" \
      --n "${GREPSEEK_SEARCH_SHARDS}"
  SEARCH_DAEMON_LOG="${GREPSEEK_SEARCH_DAEMON_LOG:-${GREPSEEK_OUTPUT_DIR}/search_daemon.log}"
  python -m inference.parallel_search.daemon \
      --socket "${GREPSEEK_SEARCH_SOCKET}" \
      --shard_dir "${GREPSEEK_SEARCH_SHARD_DIR}" \
      --corpus "${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl" \
      --timeout "${GREPSEEK_SEARCH_TIMEOUT}" \
      --max-inflight-queries "${GREPSEEK_SEARCH_MAX_INFLIGHT}" \
      --result-cache-mb "${GREPSEEK_SEARCH_RESULT_CACHE_MB}" \
      --first-stage-cache-mb "${GREPSEEK_SEARCH_FIRST_STAGE_CACHE_MB}" \
      --first-stage-max-entry-mb "${GREPSEEK_SEARCH_FIRST_STAGE_MAX_ENTRY_MB}" \
      "${DAEMON_WARMUP_ARGS[@]}" \
      --engine_log "${GREPSEEK_OUTPUT_DIR}/search_engine.jsonl" \
      --pid_file "${GREPSEEK_OUTPUT_DIR}/search_daemon.pid" \
      >"${SEARCH_DAEMON_LOG}" 2>&1 &
  SEARCH_DAEMON_PID=$!
  for _ in $(seq 1 200); do
    [[ -S "${GREPSEEK_SEARCH_SOCKET}" ]] && break
    if ! kill -0 "${SEARCH_DAEMON_PID}" 2>/dev/null; then
      cat "${SEARCH_DAEMON_LOG}" >&2 || true
      echo "error: search daemon exited before creating its socket" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [[ ! -S "${GREPSEEK_SEARCH_SOCKET}" ]]; then
    cat "${SEARCH_DAEMON_LOG}" >&2 || true
    echo "error: timed out waiting for search daemon socket" >&2
    exit 1
  fi
elif [[ "${GREPSEEK_SEARCH_ENGINE}" != "none" && "${GREPSEEK_SEARCH_ENGINE}" != "off" ]]; then
  echo "error: unsupported GREPSEEK_SEARCH_ENGINE=${GREPSEEK_SEARCH_ENGINE}" >&2
  exit 1
fi
# `trainer.experiment_name` is the name passed to verl's Tracking backend, and
# therefore the actual W&B run name (not `wandb.run_name`).  Keep the default
# from grpo_trainer.yaml unless a caller explicitly supplies one.
GREPSEEK_EXPERIMENT_NAME="${GREPSEEK_EXPERIMENT_NAME:-}"

# --- optional Weights & Biases (console-only unless BOTH a key and entity are set) ---
if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_ENTITY:-}" ]]; then
  LOGGER='[console,wandb]'; WANDB_ENABLE=true
  export WANDB_PROJECT="${WANDB_PROJECT:-grepseek_verl_grpo}"
else
  [[ -n "${WANDB_API_KEY:-}" ]] && echo "note: WANDB_API_KEY set but WANDB_ENTITY unset -> console logging only"
  LOGGER='[console]'; WANDB_ENABLE=false
fi

# These need to be individual argv entries so names with punctuation (or spaces)
# remain a single Hydra override.
TRACKING_ARGS=()
[[ -n "${GREPSEEK_EXPERIMENT_NAME}" ]] && TRACKING_ARGS+=("trainer.experiment_name=${GREPSEEK_EXPERIMENT_NAME}")
[[ "${WANDB_ENABLE}" == true ]] && TRACKING_ARGS+=("trainer.project_name=${WANDB_PROJECT}")

# --- register the agent loop (fail fast if the package isn't importable) ---
python -c "import grepseek.trainer; print('GrepSeek agent loop registered.')"

echo "============================================================"
echo "model:       ${GREPSEEK_MODEL_PATH}"
echo "train data:  ${GREPSEEK_TRAIN_FILES}"
echo "val data:    ${GREPSEEK_VAL_FILES}"
echo "corpus:      ${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl"
echo "search:      ${GREPSEEK_SEARCH_ENGINE} (${GREPSEEK_SEARCH_SHARDS} shards, max ${GREPSEEK_SEARCH_MAX_INFLIGHT} queries)"
echo "cache:       result=${GREPSEEK_SEARCH_RESULT_CACHE_MB}MiB first-stage=${GREPSEEK_SEARCH_FIRST_STAGE_CACHE_MB}MiB max-entry=${GREPSEEK_SEARCH_FIRST_STAGE_MAX_ENTRY_MB}MiB"
[[ "${GREPSEEK_SEARCH_ENGINE}" == "daemon" ]] && echo "socket:      ${GREPSEEK_SEARCH_SOCKET}"
echo "validation:  test_freq=${GREPSEEK_TEST_FREQ} val_max_samples=${GREPSEEK_VAL_MAX_SAMPLES}"
echo "output:      ${GREPSEEK_OUTPUT_DIR}"
echo "GPUs:        ${NPROC}   ulysses_sp: ${ULYSSES_SP}"
[[ -n "${GREPSEEK_EXPERIMENT_NAME}" ]] && echo "experiment:  ${GREPSEEK_EXPERIMENT_NAME}"
echo "logger:      ${LOGGER}"
echo "============================================================"

STOP_JUDGE_ARGS=()
if [[ "${GREPSEEK_STOP_JUDGE_ENABLED:-0}" == "1" ]]; then
  STOP_JUDGE_ARGS+=("+stop_judge@_global_=${GREPSEEK_STOP_JUDGE_CONFIG:-enabled}")
fi
python -m verl.trainer.main_ppo \
    --config-path "${REPO_ROOT}/rl/grepseek/trainer/config" \
    --config-name grpo_trainer \
    "${STOP_JUDGE_ARGS[@]}" \
    "hydra.searchpath=[file://${VERL_DIR}/verl/trainer/config]" \
    trainer.n_gpus_per_node="${NPROC}" \
    ray_kwargs.ray_init.num_gpus="${NPROC}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${NPROC}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    trainer.logger="${LOGGER}" \
    wandb.enable="${WANDB_ENABLE}" \
    trainer.test_freq="${GREPSEEK_TEST_FREQ}" \
    data.val_max_samples="${GREPSEEK_VAL_MAX_SAMPLES}" \
    "${TRACKING_ARGS[@]}" \
    "$@"

echo "done. checkpoints -> ${GREPSEEK_OUTPUT_DIR}"
