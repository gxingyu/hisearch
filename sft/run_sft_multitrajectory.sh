#!/usr/bin/env bash
# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
# Supervised fine-tuning on the selected multi-trajectory shell+dense data.
#
# The generated parquet keeps the complete messages and the two-tool schema;
# the student is the 9B model, while Qwen3.8-27B was used only as the teacher
# during data generation.
#
# Quick start (from the repository root):
#   bash sft/run_sft_multitrajectory.sh
#
# Every setting can be overridden through an environment variable.  The
# launcher intentionally mirrors sft/run_sft.sh, but defaults to the merged
# multi-trajectory train/validation parquet files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

if [[ ! -d "${REPO_ROOT}/sft" || ! -d "${REPO_ROOT}/rl" || ! -d "${REPO_ROOT}/inference" ]]; then
  echo "error: could not locate the grepseek repository from ${REPO_ROOT}" >&2
  exit 2
fi

root_path() {
  case "$1" in
    /*|[A-Za-z]:[\\/]*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

VERL_DIR="${VERL_DIR:-${REPO_ROOT}/verl}"
if [[ ! -d "${VERL_DIR}/verl/trainer" ]]; then
  echo "error: verl not found at ${VERL_DIR}. Restore the vendored engine from this release:" >&2
  echo "       git -C ${REPO_ROOT} checkout -- verl" >&2
  exit 1
fi
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$$}"
mkdir -p "${TRITON_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/sft/parquet}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"
TRAIN_PARQUET="$(root_path "${TRAIN_PARQUET}")"
VAL_PARQUET="$(root_path "${VAL_PARQUET}")"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/base}"
if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC="${#VISIBLE_GPUS[@]}"
  else
    NPROC="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
  fi
fi
NPROC="${NPROC//[[:space:]]/}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/sft/hisearch-multitrajectory-$(date +%Y%m%d-%H%M%S)}"
SAVE_DIR="$(root_path "${SAVE_DIR}")"

MAX_LENGTH="${MAX_LENGTH:-16384}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"
ULYSSES_SP_SIZE="${ULYSSES_SP_SIZE:-${NPROC}}"
LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-after_each_epoch}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ ]] || (( NPROC < 2 )); then
  echo "error: set NPROC to at least 2 GPUs for this FSDP launcher" >&2
  exit 1
fi
if [[ ! "${ULYSSES_SP_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( NPROC % ULYSSES_SP_SIZE != 0 )); then
  echo "error: ULYSSES_SP_SIZE must be a positive divisor of NPROC" >&2
  exit 1
fi

if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_ENTITY:-}" ]]; then
  LOGGER='[console,wandb]'
  export WANDB_PROJECT="${WANDB_PROJECT:-grepseek-sft-multitrajectory}"
else
  [[ -n "${WANDB_API_KEY:-}" ]] && echo "note: WANDB_API_KEY set but WANDB_ENTITY unset -> console logging only" >&2
  LOGGER='[console]'
fi

[[ -f "${TRAIN_PARQUET}" ]] || { echo "missing TRAIN_PARQUET: ${TRAIN_PARQUET}" >&2; exit 1; }
[[ -f "${VAL_PARQUET}" ]] || { echo "missing VAL_PARQUET: ${VAL_PARQUET}" >&2; exit 1; }
if [[ -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "error: local MODEL_PATH has no config.json: ${MODEL_PATH}" >&2
  exit 1
fi
mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "model:       ${MODEL_PATH}"
echo "train data:  ${TRAIN_PARQUET}"
echo "validation:  ${VAL_PARQUET}"
echo "checkpoints: ${SAVE_DIR}"
echo "GPUs:        ${NPROC}   ulysses_sp: ${ULYSSES_SP_SIZE}"
echo "max_length:  ${MAX_LENGTH}   global bs: ${TRAIN_BATCH_SIZE}   epochs: ${TOTAL_EPOCHS}"
echo "test freq:   ${TEST_FREQ}"
echo "logger:      ${LOGGER}"
echo "============================================================"

cd "${REPO_ROOT}"
torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.sft_trainer \
        data.train_files="${TRAIN_PARQUET}" \
        data.val_files="${VAL_PARQUET}" \
        data.messages_key=messages \
        data.tools_key=tools \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.micro_batch_size_per_gpu="${MICRO_BATCH_PER_GPU}" \
        data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
        data.use_dynamic_bsz=true \
        data.max_length="${MAX_LENGTH}" \
        data.truncation=right \
        data.pad_mode=no_padding \
        data.ignore_input_ids_mismatch=true \
        model.path="${MODEL_PATH}" \
        model.use_remove_padding=true \
        model.enable_gradient_checkpointing=true \
        model.enable_activation_offload=true \
        engine.strategy=fsdp \
        engine.dtype=bfloat16 \
        engine.param_offload=true \
        engine.optimizer_offload=true \
        engine.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
        optim.lr="${LR}" \
        optim.weight_decay="${WEIGHT_DECAY}" \
        optim.lr_warmup_steps_ratio="${WARMUP_RATIO}" \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.logger="${LOGGER}" \
        trainer.project_name="${WANDB_PROJECT:-gsm8k-sft}" \
        trainer.experiment_name="$(basename "${SAVE_DIR}")" \
        trainer.default_local_dir="${SAVE_DIR}" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${NPROC}" \
        trainer.resume_mode=auto \
        ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}}

echo "done. checkpoints -> ${SAVE_DIR}"
