# Multi-trajectory hybrid SFT

The generator executes real dense and shell tools against the same corpus,
audits the resulting trajectories, and exports training messages with both
schemas. Generation, export, and policy training are separate commands.

Install [the policy environment](../TRAINING_ENV.md) and the dependencies in
`sft/data_generation/requirements.txt`. Prepare [retrieval](../retrieval/README.md)
and configure a teacher endpoint via `LLM_BASE_URL`, `LLM_MODEL`, and
`OPENAI_API_KEY` in the environment. The teacher must support structured JSON
responses; use a served model name supported by your endpoint.

```bash
python sft/data_generation/create_hybrid_data.py \
  --tokenizer models/base --dataset hotpotqa --split train --n 200 \
  --out-dir outputs/hybrid_sft
python sft/data_generation/to_parquet.py \
  --in outputs/hybrid_sft/sft.jsonl --out_dir data/sft/parquet --include_tools
MODEL_PATH="$PWD/models/base" DATA_DIR="$PWD/data/sft/parquet" NPROC=8 \
  bash sft/run_sft_multitrajectory.sh
```

`--input` accepts local training QA rows with `id`, `question`, and
`golden_answers`. Default strategies are free choice, dense first, shell first,
shell only, and dense only; each uses the same five-call budget and 1,024-token
observations. Teacher-only sampling constraints and backward references do not
enter exported policy messages. No per-question tool-mixture quota is imposed.

`traces.jsonl` contains all attempts and review outcomes; `sft.jsonl` contains
selected trajectories; `summary.json` distinguishes generated and retained
examples. These counts are synthesis statistics, not independent model scores.
`to_parquet.py` keeps all trajectories for a question in the same split.

Filtering combines answer EM, literal quotes from visible observations,
chronological evidence/query checks, relation coverage, and observation
ablations. It cannot guarantee the absence of every unsupported entity or
alias. Inspect evidence and rejected cases before training or releasing data.
`--resume` verifies run/question/generator fingerprints; use a fresh output
directory for a changed configuration.

The SFT launcher defaults to batch size 32, one epoch, learning rate `5e-6`,
maximum length 16,384, and a GPU count inferred from `CUDA_VISIBLE_DEVICES` or
`NPROC`. `ULYSSES_SP_SIZE` must divide `NPROC`, which must be at least 2.
The actual number of updates depends on data size and batching; no fixed step
count is promised by this source release.

Merge a saved SFT FSDP checkpoint into a local model for hybrid RL:

```bash
export PYTHONPATH="$PWD/verl:${PYTHONPATH:-}"
python -m verl.model_merger merge --backend fsdp \
  --local_dir /absolute/path/to/sft/global_step_N \
  --target_dir "$PWD/models/sft" --trust-remote-code
```

Retained legacy shell-generation utilities support the shared helper imports;
`create_hybrid_data.py` is the documented hybrid entry point.
