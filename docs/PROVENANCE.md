# Source provenance and release changes

HiSearch derives from [GrepSeek](https://github.com/alirezasalemi7/grepseek),
whose research checkout is based on commit
`1f6ea58` (`[feature] more container based instruction`). This release includes
the current local implementation, including changes that had not been committed
to that upstream checkout. It starts a new source-only Git history so local
experiment files and any old credentials are not carried into the new remote.

## Implementation retained

- Shared dense/shell retrieval schemas, fixed dense top-3, original-byte corpus
  lookup and rollout-scoped candidate handles.
- Sharded direct-corpus execution with caching, coalescing, bounded output,
  timeout cleanup, and isolated tool executors.
- Executed forward teacher trajectories, shared-prefix branching, visible
  evidence/chronology review, grouped SFT export, and SFT training launcher.
- Hybrid policy dataset, agent loop, reward, configuration, and prompts.
- Five maintained process rubrics: visible-event scoring, group fallback/masking,
  pre-filter statistics, reference calibration, one-item maintenance, versioning,
  offline preparation/replay, and checkpoint save/restore.
- Local verl changes in agent/reward parsing, Qwen3.5 handling, dynamic sampling,
  actor/worker training, tracking, checkpointing, and trainer configuration.

These describe implemented source paths, not a claim that all features have
independently measured training gains. See [release_status.md](release_status.md).

## Publication changes

The release copies source files from the current implementation and retains
the vendored runtime. Only the publishing copy receives portable path defaults,
conditional environment activation, local endpoint defaults, optional W&B
credentials, and public documentation. Policy loss, the five-criterion reward
contract, and maintenance thresholds are carried over from the research source.
Modified upstream Python, shell, and YAML files carry a HiSearch modification
notice. Existing copyright/license headers are retained. Two unit-test fixtures
were aligned with the portable experiment name and the existing agent's
`max_user_turns` field; the production training logic was unchanged.

Excluded resources include corpus/model/index files, generated datasets,
checkpoints, rollout and W&B logs, environment caches, binary assets, old
research diaries, one-off orchestration dependent on unpublished experiment
files, machine-specific launchers, and upstream development material that is
not required by the runtime. `rl/grepseek/data/` is a Python source package and
is retained despite the root `/data/` artifact exclusion.

Original GrepSeek citation metadata is preserved unchanged in
[GrepSeek.CITATION.cff](GrepSeek.CITATION.cff). No author list or paper citation
for the new project is guessed. Search-R1 is installed separately; its encoder
and public resources keep their own licenses and attribution. This repository
does not redistribute third-party models, datasets, or indexes.
