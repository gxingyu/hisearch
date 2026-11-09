# GSPO with five maintained process rubrics

`run_gspo_hybrid_rubric.sh` adds exactly `0.1` for each passed item to raw
token-F1. Valid rewards range from 0 to 1.5; invalid/failed answers keep their
baseline handling and receive no process bonus. The GSPO loss and response loss
mask are unchanged. `algorithm.rubric.enabled` defaults to false in the shared
configuration, so other launchers retain their existing behavior.

```bash
# No network, services, policy rollout, or training: inspect local inputs.
bash run_gspo_hybrid_rubric.sh --check-only

# CPU tests with an in-process mock judge, including trainer/refill/checkpoint tests.
bash run_gspo_hybrid_rubric.sh --self-test
```

No arguments also means `--check-only`. Only explicit `--train` invokes the
existing hybrid launcher. The new launcher defaults to a local merged model at `models/sft`, 256 question groups, existing rollout count (5), shared five-call
limit, and the existing GSPO schedule. It disables optional length/efficiency/
stop-control additions to keep the scalar reward precisely F1 plus rubric scores.
It starts no judge and stops no existing services. Its downstream hybrid launcher
has the usual retrieval-service preflight when training is explicitly requested.

The bundled `initial_rubrics.json` is a **human-authored bootstrap**, not a claim
of offline model generation. To generate a model-written initial version and a
separate training-only reference panel using an already-running judge:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/rl:$PWD/verl:${PYTHONPATH:-}"
export GREPSEEK_RUBRIC_URL=http://127.0.0.1:8100/v1
export GREPSEEK_RUBRIC_MODEL=qwen38-27b-stop-judge
export GREPSEEK_RUBRIC_API_KEY=local-no-auth

python rl/rubric_workflow.py initialize \
  --train-file data/rl/nq_hotpot/train.jsonl \
  --output outputs/rubric_setup/initial.json

# Read an existing training rollout file; no model or tool is executed here.
python rl/rubric_workflow.py export-inputs \
  --train-file data/rl/nq_hotpot/train.jsonl \
  --input /path/to/existing/train_rollouts/1.jsonl \
  --limit 256 --output outputs/rubric_setup/reference_inputs.jsonl

python rl/rubric_workflow.py prepare-calibration \
  --initial outputs/rubric_setup/initial.json \
  --input outputs/rubric_setup/reference_inputs.jsonl \
  --output outputs/rubric_setup/reference.jsonl

export GREPSEEK_RUBRIC_INITIAL_PATH="$PWD/outputs/rubric_setup/initial.json"
export GREPSEEK_RUBRIC_CALIBRATION_PATH="$PWD/outputs/rubric_setup/reference.jsonl"
bash run_gspo_hybrid_rubric.sh --check-only --require-calibration

# Run only when policy training is intended; not part of offline validation.
bash run_gspo_hybrid_rubric.sh --train
```

The initializer samples training questions with seed 42 and records their ids and
hash. The reference preparer uses initial-rubric model judgments, explicitly marked
as **model labels, not human ground truth**; independent review or human-corrected
references are preferable before interpreting semantic accuracy. For each item,
publication requires at least eight clearly labeled reference questions, including
both pass and fail examples. Proposal cases always exclude all reference qids.
New references must be prepared after changing the initial snapshot, because each
row is bound to its initial snapshot hash.

Reference JSONL rows contain `split: "train"`, `qid`, `trajectory_id`, `question`,
`events` (`id`, `kind`, `text`), `initial_snapshot_hash`, `reference_source`, and
`reference_statuses` mapping all five ids to `pass`, `fail`, `not_applicable` or
`uncertain`. The archive exporter joins question ids against the supplied training
file and uses only text from saved outputs; it marks this weaker provenance as
decoded archival text rather than pretending to recover original token spans.
New live rubric runs instead export events from the actual final response token
ids and masks, after tool/output/context cutoffs, including environment prefills.

One judge request scores all five criteria without reference answers, F1, or hidden
backend text. Quotes must exactly match visible events. Technical failures retry
then fall back to outcome rewards for the whole question group; semantic uncertain
or inapplicable items are masked consistently across that group's valid members.
Neither status enters pass/fail denominators. Invalid answers do not force a
technical fallback for the valid members of their group.

The driver freezes a snapshot over the entire outer step, including every refill.
Statistics include all sampled groups before filtering and separately record the
retained batch. Rates balance questions; discrimination measures pass/fail
differences **within** question groups. Mature triggers inspect saturation,
ambiguity, high failure rates and lack of within-group distinction. High failure
does not automatically delete a valid capability target. Fully uncertain criteria
can mature using applicable question counts even with zero clear pass/fail labels.
Maintenance pools at most 20 recent steps of matching item revisions, so sparse
scenarios can accumulate evidence; per-step statistics are still reported
separately. A revised item's previous labels never enter its new-version rates.
The compact evidence window is included in checkpoints.
If dynamic collection drops an incomplete batch at an epoch boundary, its raw
records are archived as an aborted attempt, without advancing rubric versions or
evidence counters. The next epoch can retry the same outer step.

After an actor update, upkeep can return KEEP or revise/replace one item. A
candidate must preserve five items and fixed rewards, cite real proposal cases,
pass a separate semantic review, and meet reference agreement/false-penalty gates.
It takes effect at the next step; revised items start new statistics. Missing or
rejected calibration preserves the old version. The `--train` preflight requires
a usable reference panel to avoid silently advertising a static run as dynamic.

Audit files live under `<run>/rubric/`: immutable `versions/`, per-step
`used_snapshot.json`, all-sampled trajectory/grade JSONL, `summary.json`, request
usage/errors, and current state. Each policy checkpoint also stores
`rubric_state.json`; resume requires matching completed step, initial snapshot,
reference panel, judge configuration and maintenance configuration. Use the underlying trainer's resume
configuration for continuation; this fresh-run launcher deliberately uses
`trainer.resume_mode=disable` and refuses an existing output directory.

For an offline replay (judge requests only; no actor or optimizer), use
`python rl/rubric_workflow.py replay --initial ... --calibration ... --input ...
--output ...`. Input rows add `step`, `group_id`, `valid`, `f1`, `base_score`, and
optional `retained` to the visible-event format. Steps start at 1 and are
consecutive. Tests and replay validate plumbing; they do not establish judge
semantic accuracy or RL gains. A 0.5 process bonus can reorder trajectories with
different partial F1, and group normalization can amplify process-score noise;
always evaluate raw EM/F1 independently of the evolving reward.
