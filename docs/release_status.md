# Source release status

Prepared on 2026-11-09 (Asia/Shanghai, UTC+8).

| Component | Evidence available | Remaining limits |
| --- | --- | --- |
| Shared hybrid retrieval | Existing implementation and local unit tests for scopes, isolation, tools, and corpus alignment | No full-corpus GPU retrieval benchmark rerun for publication |
| Multi-trajectory SFT | Generator/export/training source and historical SFT runs in the research workspace | Generated data and policy weights are not included; no retraining during publication |
| Outcome-only GSPO | Existing hybrid training path and historical research runs | No new benchmark table or matched tool ablation in this release |
| Five-rubric GSPO | CPU/mock scoring, maintenance, refill, trainer, and checkpoint integration tests | Real model-written initialization, judge semantic calibration/throughput, and RL gains remain unverified |

The bundled `initial_rubrics.json` identifies itself as a human-authored
bootstrap template. Reference panels produced by `prepare-calibration` are
explicitly model labels rather than human ground truth. Review them before
interpreting rubric agreement or training effects.

This release includes source, prompts, configurations, small metadata, and tests.
All large training resources and experiment evidence remain outside Git. No
new policy rollout, optimizer update, checkpoint, or service startup is needed
to prepare or check the source release.

## Publication verification

Checks completed in the independent release checkout on 2026-11-09 (UTC+8):

- 209 RL unit/integration tests passed, including rubric scoring, maintenance,
  refill, checkpoint recovery, tool protocols, and launcher mocks.
- 91 SFT generation/export/audit tests passed.
- 35 sharded shell execution/cache/daemon tests passed on synthetic corpora.
- All 532 Python files parsed; all included Bash scripts passed `bash -n`.
- A fresh Python 3.11 environment installed only `requirements-offline.txt`:
  the documented check-only command and 35 lightweight rubric tests passed
  without Torch, Ray, or vLLM installed.
- The tracked-file audit passed: 656 files, about 6.3 MB total, with no file
  exceeding 1 MiB and no matches for the checked credential patterns. Local
  documentation links and the scan for private machine defaults passed.
- AST comparison of 12 central retrieval/SFT/rubric/training files confirmed
  unchanged executable code relative to the research snapshot. Publication
  changes are documented in [PROVENANCE.md](PROVENANCE.md).

The full regression suite used the existing policy environment described in
[TRAINING_ENV.md](../TRAINING_ENV.md). The fresh lightweight check used Python
3.11.11, httpx 0.28.1, and NumPy 2.4.6. These results establish source packaging
and offline functionality; they do not establish real judge accuracy, full GPU
setup reproducibility, or policy-training gains.

To reproduce the full CPU/mock suites in the policy environment:

```bash
export PYTHONPATH="$PWD:$PWD/rl:$PWD/verl:${PYTHONPATH:-}"
python -m unittest discover -s rl/tests -p 'test_*.py'
python -m unittest discover -s sft/data_generation/tests -p 'test_*.py'
python -m unittest discover -s inference/parallel_search/tests -p 'test_*.py'
python scripts/check_release.py
```
