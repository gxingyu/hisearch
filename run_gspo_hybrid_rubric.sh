#!/usr/bin/env bash
# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
# F1 + five binary rubric bonuses. No arguments means local checks only.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rl:${SCRIPT_DIR}/verl:${PYTHONPATH:-}"
export GREPSEEK_ROOT="${SCRIPT_DIR}"
export GREPSEEK_RUBRIC_INITIAL_PATH="${GREPSEEK_RUBRIC_INITIAL_PATH:-${SCRIPT_DIR}/rl/grepseek/trainer/config/rubric/initial_rubrics.json}"
export GREPSEEK_RUBRIC_CALIBRATION_PATH="${GREPSEEK_RUBRIC_CALIBRATION_PATH:-}"
export GREPSEEK_RUBRIC_URL="${GREPSEEK_RUBRIC_URL:-http://127.0.0.1:8100/v1}"
export GREPSEEK_RUBRIC_MODEL="${GREPSEEK_RUBRIC_MODEL:-qwen38-27b-stop-judge}"
MODE="${1:---check-only}"
if (( $# )); then shift; fi
case "${MODE}" in
  --check-only)
    exec python rl/rubric_workflow.py check \
      --initial "${GREPSEEK_RUBRIC_INITIAL_PATH}" \
      --calibration "${GREPSEEK_RUBRIC_CALIBRATION_PATH}" "$@"
    ;;
  --self-test)
    exec python -m unittest rl.tests.test_rubric_workflow rl.tests.test_rubric_integration rl.tests.test_rubric_launcher rl.tests.test_rubric_cli -v "$@"
    ;;
  --train) ;;
  *) echo 'usage: run_gspo_hybrid_rubric.sh [--check-only | --self-test | --train [Hydra overrides...]]' >&2; exit 2 ;;
esac

# Require an auditable training reference panel before enabling automatic changes.
# This preflight only contacts an already-running judge; it starts no judge.
export GREPSEEK_RUBRIC_API_KEY="${GREPSEEK_RUBRIC_API_KEY:-local-no-auth}"
python rl/rubric_workflow.py check --initial "${GREPSEEK_RUBRIC_INITIAL_PATH}" \
  --calibration "${GREPSEEK_RUBRIC_CALIBRATION_PATH}" --require-calibration --live
export GREPSEEK_MODEL_PATH="${GREPSEEK_MODEL_PATH:-${SCRIPT_DIR}/models/sft}"
export GREPSEEK_EXPERIMENT_NAME="${GREPSEEK_EXPERIMENT_NAME:-hisearch-gspo-rubric-$(date +%Y%m%d-%H%M%S)}"
export GREPSEEK_EXPERIMENT_NAME_EXPLICIT=1
export GREPSEEK_OUTPUT_DIR="${GREPSEEK_OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/rl/${GREPSEEK_EXPERIMENT_NAME}}"
unset GREPSEEK_STOP_JUDGE_ENABLED GREPSEEK_WANDB_RUN_ID GREPSEEK_WANDB_RUN_NAME GREPSEEK_RESUME_FROM_PATH WANDB_RUN_ID WANDB_RESUME

# Keep the reward contract enforced after caller overrides; maintained thresholds
# and rollout.n may still be configured explicitly for controlled experiments.
exec bash "${SCRIPT_DIR}/run_gspo_hybrid.sh" "$@" \
  algorithm.rubric.enabled=true algorithm.rubric.per_item_reward=0.1 \
  "algorithm.rubric.initial_path=${GREPSEEK_RUBRIC_INITIAL_PATH}" \
  "algorithm.rubric.calibration_path=${GREPSEEK_RUBRIC_CALIBRATION_PATH}" \
  "algorithm.rubric.output_dir=${GREPSEEK_OUTPUT_DIR}/rubric" \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false \
  algorithm.filter_groups.enable=true algorithm.filter_groups.metric=seq_reward \
  algorithm.stop_control.enabled=false algorithm.answer_body_gate.enabled=false algorithm.route_efficiency.enabled=false \
  custom_reward_function.name=grepseek_reward custom_reward_function.reward_kwargs.reward_metric=f1 \
  custom_reward_function.reward_kwargs.enable_length_decay=false custom_reward_function.reward_kwargs.enable_penalty_warmup=false \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.rollout.agent.default_agent_loop=grepseek_hybrid_tool_limit_invalid_agent \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${SCRIPT_DIR}/rl/grepseek/trainer/config/agent_loops_hybrid_tool_limit_invalid.yaml" \
  data.train_batch_size=256 trainer.critic_warmup=0 trainer.resume_mode=disable
