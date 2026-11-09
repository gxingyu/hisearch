#!/usr/bin/env bash
# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/rl:${SCRIPT_DIR}/verl:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GREPSEEK_ROOT="${SCRIPT_DIR}"
export GREPSEEK_MODEL_PATH="${GREPSEEK_MODEL_PATH:-${SCRIPT_DIR}/models/sft}"
export GREPSEEK_TRAIN_FILES="${GREPSEEK_TRAIN_FILES:-data/rl/nq_hotpot/train.jsonl}"
export GREPSEEK_VAL_FILES="${GREPSEEK_VAL_FILES:-data/rl/flashrag_7bench/val.jsonl}"
export GREPSEEK_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT:-${SCRIPT_DIR}/data/wiki_18_corpus}"
export GREPSEEK_DENSE_DATA_ROOT="${GREPSEEK_DENSE_DATA_ROOT:-${SCRIPT_DIR}/data/search_r1}"
export GREPSEEK_DENSE_CORPUS_PATH="${GREPSEEK_DENSE_CORPUS_PATH:-${SCRIPT_DIR}/data/wiki_18_corpus/wiki_corpus.jsonl}"
export GREPSEEK_DENSE_URL="${GREPSEEK_DENSE_URL:-http://127.0.0.1:8000/retrieve}"
export GREPSEEK_DENSE_BATCH_SIZE="${GREPSEEK_DENSE_BATCH_SIZE:-32}"
export GREPSEEK_DENSE_BATCH_WAIT_MS="${GREPSEEK_DENSE_BATCH_WAIT_MS:-10}"
export GREPSEEK_DENSE_QUEUE_SIZE="${GREPSEEK_DENSE_QUEUE_SIZE:-4096}"
export WANDB_PROJECT="${WANDB_PROJECT:-hisearch-rl}"
if [[ "${GREPSEEK_EXPERIMENT_NAME_EXPLICIT:-0}" == "1" && -n "${GREPSEEK_EXPERIMENT_NAME:-}" ]]; then
  : "Using explicitly requested experiment name: ${GREPSEEK_EXPERIMENT_NAME}"
else
  unset GREPSEEK_EXPERIMENT_NAME GREPSEEK_OUTPUT_DIR
  GREPSEEK_EXPERIMENT_NAME="hisearch-gspo-hybrid-$(date +%Y%m%d-%H%M%S)"
fi
export GREPSEEK_EXPERIMENT_NAME
export GREPSEEK_OUTPUT_DIR="${GREPSEEK_OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/rl/${GREPSEEK_EXPERIMENT_NAME}}"

CHECK_ONLY=false
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=true
  shift
fi
if [[ -e "${GREPSEEK_OUTPUT_DIR}" || -L "${GREPSEEK_OUTPUT_DIR}" ]]; then
  echo "error: experiment output already exists: ${GREPSEEK_OUTPUT_DIR}" >&2
  exit 2
fi
SETUP_DIR="${SCRIPT_DIR}/outputs/hybrid_startup/${GREPSEEK_EXPERIMENT_NAME}"
mkdir -p "${SETUP_DIR}"

echo "Checking the SFT checkpoint, QA prompts and corpus snapshot..."
python -m inference.retrieval_env.preflight --local-only >"${SETUP_DIR}/local_setup.json"
cat "${SETUP_DIR}/local_setup.json"
if [[ "${CHECK_ONLY}" == true ]]; then
  echo "Local checks passed. No retrieval service or training was started."
  exit 0
fi

DENSE_PID=""
cleanup_dense_service() {
  if [[ -n "${DENSE_PID}" ]] && kill -0 "${DENSE_PID}" 2>/dev/null; then
    kill "${DENSE_PID}" 2>/dev/null || true
    wait "${DENSE_PID}" 2>/dev/null || true
  fi
}
trap cleanup_dense_service EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if python -m inference.retrieval_env.preflight --health-only --timeout 5 >"${SETUP_DIR}/health.json" 2>/dev/null; then
  echo "Checking existing dense retrieval service at ${GREPSEEK_DENSE_URL}"
else
  mapfile -t DENSE_BIND < <(python -c 'import os; from urllib.parse import urlsplit; parsed=urlsplit(os.environ["GREPSEEK_DENSE_URL"]); print(parsed.scheme); print(parsed.hostname); print(parsed.port or 8000)')
  if [[ "${DENSE_BIND[0]}" != "http" || ( "${DENSE_BIND[1]}" != "127.0.0.1" && "${DENSE_BIND[1]}" != "localhost" ) ]]; then
    echo "error: remote dense retrieval service is unavailable: ${GREPSEEK_DENSE_URL}" >&2
    exit 2
  fi
  echo "Starting dense retrieval on GPUs ${GREPSEEK_DENSE_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES}} (log: ${SETUP_DIR}/dense_server.log)"
  CUDA_VISIBLE_DEVICES="${GREPSEEK_DENSE_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES}}" \
    bash "${SCRIPT_DIR}/run_search_r1.sh" \
      --corpus "${GREPSEEK_DENSE_CORPUS_PATH}" \
      --data-root "${GREPSEEK_DENSE_DATA_ROOT}" \
      --host "${DENSE_BIND[1]}" --port "${DENSE_BIND[2]}" \
      --batch-size "${GREPSEEK_DENSE_BATCH_SIZE}" \
      --batch-wait-ms "${GREPSEEK_DENSE_BATCH_WAIT_MS}" \
      --queue-size "${GREPSEEK_DENSE_QUEUE_SIZE}" \
      >"${SETUP_DIR}/dense_server.log" 2>&1 &
  DENSE_PID=$!
  printf '%s\n' "${DENSE_PID}" >"${SETUP_DIR}/dense_server.pid"
  DENSE_WAIT_START=${SECONDS}
  until python -m inference.retrieval_env.preflight --health-only --timeout 3 >"${SETUP_DIR}/health.json" 2>/dev/null; do
    if ! kill -0 "${DENSE_PID}" 2>/dev/null; then
      tail -n 60 "${SETUP_DIR}/dense_server.log" >&2
      echo "error: dense retrieval service exited before becoming ready" >&2
      exit 2
    fi
    if (( SECONDS - DENSE_WAIT_START >= ${GREPSEEK_DENSE_STARTUP_TIMEOUT:-900} )); then
      tail -n 60 "${SETUP_DIR}/dense_server.log" >&2
      echo "error: timed out waiting for dense retrieval service" >&2
      exit 2
    fi
    sleep 2
  done
fi

if ! python - "${SETUP_DIR}/health.json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as health_file:
    health = json.load(health_file)

expected = {
    "parallel_shards": True,
    "batch_size": int(os.environ["GREPSEEK_DENSE_BATCH_SIZE"]),
    "batch_wait_ms": float(os.environ["GREPSEEK_DENSE_BATCH_WAIT_MS"]),
}
for key, value in expected.items():
    if health.get(key) != value:
        raise SystemExit(f"dense service did not report the requested optimized setting: {key}={health.get(key)!r}, expected {value!r}")
PY
then
  echo "error: ${GREPSEEK_DENSE_URL} does not match the optimized batched/parallel settings; restart it with the requested settings before training" >&2
  cat "${SETUP_DIR}/health.json" >&2
  exit 2
fi
echo "Optimized dense retrieval ready (batch=${GREPSEEK_DENSE_BATCH_SIZE}, wait=${GREPSEEK_DENSE_BATCH_WAIT_MS}ms, parallel shards)."

python -m inference.retrieval_env.preflight --service-only >"${SETUP_DIR}/service_setup.json"
cat "${SETUP_DIR}/service_setup.json"
echo "Starting GSPO with shell + dense_search (fixed topk=3)."
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B experiment: ${GREPSEEK_EXPERIMENT_NAME}"
echo "Training log: ${GREPSEEK_OUTPUT_DIR}/train.log"

bash "${SCRIPT_DIR}/run_gspo.sh" \
  "data.custom_cls.path=${SCRIPT_DIR}/rl/grepseek/trainer/verl_integration/hybrid_dataset.py" \
  data.custom_cls.name=HybridDataset \
  "data.system_prompt_path=${SCRIPT_DIR}/rl/prompts/system_prompt_hybrid.txt" \
  "data.tool_spec_template_path=${SCRIPT_DIR}/rl/prompts/tool_spec_template_hybrid.txt" \
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=${SCRIPT_DIR}/rl/grepseek/trainer/config/tools/search_tools_hybrid.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=grepseek_hybrid_agent \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${SCRIPT_DIR}/rl/grepseek/trainer/config/agent_loops_hybrid.yaml" \
  "$@"
