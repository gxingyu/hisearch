# Hybrid GSPO training

Run from the repository root after installing [the training stack](../TRAINING_ENV.md).
The `grepseek` package name is retained from the upstream project.

## Inputs and outcome reward

`prepare_rl_data.py --out_dir data/rl/nq_hotpot` downloads NQ and HotpotQA
training/dev questions from `RUC-NLPIR/FlashRAG_datasets`. Rows contain `id`,
`question`, `golden_answers`, and `source`. The dataset loader supports multiple
gold answers; the outcome reward uses answer token-F1 and strict format checks.

For the seven-benchmark validation builder, download the FlashRAG dataset
snapshot separately into `data/flashrag_datasets` and run:

```bash
python rl/prepare_flashrag_val.py --dataset-root data/flashrag_datasets \
  --output data/rl/flashrag_7bench/val.jsonl
```

The builder checks its source splits, samples up to 500 examples per larger
dataset and retains Bamboogle's 125 examples. This 3,125-question mix differs
from the training-source dev set. Inspect the generated `build_summary.json`.

`run_gspo_hybrid.sh` performs local model/prompt/corpus checks and verifies the
dense service settings before invoking `run_gspo.sh` and `rl/run_rl.sh`.
`--check-only` on the hybrid launcher checks local inputs; ordinary invocation
starts training and may start a local dense service if none is available.
Model, data, corpus, service, GPU count, and output paths are configurable through
the environment variables shown in the [root README](../README.md).

GRPO group advantages feed the existing GSPO sequence-level loss. The baseline
uses five rollouts per question, dynamically filters groups with flat rewards,
and refills until it has 256 retained question groups. Each outer update can
therefore sample more than 1,280 trajectories. Tool observations are masked from
the policy loss, and the dense encoder/index are frozen.

The original hybrid baseline and the rubric recipe use different default
over-limit behavior. To run an outcome-only control with the rubric recipe's
five-call/over-limit-invalid protocol:

```bash
bash run_gspo_hybrid.sh \
  actor_rollout_ref.rollout.agent.default_agent_loop=grepseek_hybrid_tool_limit_invalid_agent \
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=$PWD/rl/grepseek/trainer/config/agent_loops_hybrid_tool_limit_invalid.yaml" \
  algorithm.rubric.enabled=false trainer.val_before_train=true
```

Use the same policy checkpoint, prompt, datasets, budgets, seeds, hardware, and
remaining settings for comparisons. Optional historical gates, route/length
rewards, and prefix stop judges remain in the shared runtime but are disabled
by the rubric recipe.

## Rubric-based RL

See [docs/rubric_rl.md](../docs/rubric_rl.md) for initial criteria, reference
panels, scoring, maintenance, replay, and checkpoint recovery. The launcher
defaults to local checks and requires explicit `--train` plus a usable training
reference panel to enable dynamic maintenance.

## Evaluation and checkpoint export

Use the same hybrid launcher and agent configuration with
`trainer.val_only=true trainer.val_before_train=true` and point
`GREPSEEK_MODEL_PATH` at a local merged model. Choose a fresh output directory.
This still requires the training runtime and retrieval services, but performs
validation without an optimizer update. Do not use the legacy shell-only
`inference/run.py` as the hybrid evaluation entry point.

RL checkpoints contain actor FSDP shards, tokenizer/config, and, when enabled,
`rubric_state.json`. Export policy weights with:

```bash
CKPT_DIR=/absolute/path/to/global_step_N HF_DIR="$PWD/models/rl" \
  bash rl/convert_to_hf.sh
```

Merged weights are for inference or fresh initialization. Exact training resume
requires the original distributed checkpoint, optimizer state, matching rubric
state, reference panel, and judge/maintenance configuration. The fresh-run
rubric launcher sets `trainer.resume_mode=disable`; use the underlying trainer
configuration for an intentional continuation.
