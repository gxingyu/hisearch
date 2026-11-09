#!/usr/bin/env bash
# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Optional W&B credentials are read only from the process environment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
export ULYSSES_SP="${ULYSSES_SP:-2}"

export GREPSEEK_MODEL_PATH="${GREPSEEK_MODEL_PATH:-${SCRIPT_DIR}/models/sft}"
if [[ ! -f "${GREPSEEK_MODEL_PATH}/config.json" ]]; then
  echo "error: local SFT starting model missing: ${GREPSEEK_MODEL_PATH}" >&2
  exit 2
fi
export GREPSEEK_TRAIN_FILES="${GREPSEEK_TRAIN_FILES:-data/rl/nq_hotpot/train.jsonl}"
export GREPSEEK_VAL_FILES="${GREPSEEK_VAL_FILES:-data/rl/flashrag_7bench/val.jsonl}"

export GREPSEEK_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT:-${SCRIPT_DIR}/data/wiki_18_corpus}"
export GREPSEEK_SEARCH_ENGINE="${GREPSEEK_SEARCH_ENGINE:-daemon}"
export GREPSEEK_SEARCH_SHARDS="${GREPSEEK_SEARCH_SHARDS:-32}"
export GREPSEEK_SEARCH_SHARD_DIR="${GREPSEEK_SEARCH_SHARD_DIR:-${GREPSEEK_CORPUS_ROOT}/.grepseek_shards_${GREPSEEK_SEARCH_SHARDS}}"
export GREPSEEK_SEARCH_MAX_INFLIGHT="${GREPSEEK_SEARCH_MAX_INFLIGHT:-4}"
export GREPSEEK_SEARCH_RESULT_CACHE_MB="${GREPSEEK_SEARCH_RESULT_CACHE_MB:-512}"
export GREPSEEK_SEARCH_FIRST_STAGE_CACHE_MB="${GREPSEEK_SEARCH_FIRST_STAGE_CACHE_MB:-4096}"
export GREPSEEK_SEARCH_FIRST_STAGE_MAX_ENTRY_MB="${GREPSEEK_SEARCH_FIRST_STAGE_MAX_ENTRY_MB:-64}"

export GREPSEEK_TEST_FREQ=40
export GREPSEEK_VAL_MAX_SAMPLES="${GREPSEEK_VAL_MAX_SAMPLES:--1}"

export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-hisearch-rl}"

unset WANDB_RUN_ID WANDB_RESUME
export GREPSEEK_EXPERIMENT_NAME="${GREPSEEK_EXPERIMENT_NAME:-hisearch-gspo-$(date +%Y%m%d-%H%M%S)}"
export GREPSEEK_OUTPUT_DIR="${GREPSEEK_OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/rl/${GREPSEEK_EXPERIMENT_NAME}}"
if [[ -e "${GREPSEEK_OUTPUT_DIR}" || -L "${GREPSEEK_OUTPUT_DIR}" ]]; then
  echo "error: experiment output already exists: ${GREPSEEK_OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${GREPSEEK_OUTPUT_DIR}"

RAY_ADDRESS="${RAY_ADDRESS:-local}" bash rl/run_rl.sh \
  trainer.total_training_steps=200 \
  trainer.test_freq=40 \
  trainer.save_freq=40 \
  +trainer.keep_latest_and_best_checkpoint=true \
  trainer.best_checkpoint.enable=true \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.clip_ratio_low=0.20 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  custom_reward_function.reward_kwargs.format_penalty=1.0 \
  algorithm.filter_groups.enable=true \
  ray_kwargs.ray_init.num_cpus=32 \
  "$@" 2>&1 | tee -a "${GREPSEEK_OUTPUT_DIR}/train.log"
