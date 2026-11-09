# HiSearch release progress

## Current state — 2026-11-09, Asia/Shanghai (UTC+8)

Source-only publication of the current hybrid retrieval + multi-trajectory SFT
+ five-rubric GSPO implementation. The full research diary and large experiment
artifacts are intentionally excluded from this public repository.

Portable launchers and English/Chinese documentation are prepared for the
initial source release at `https://github.com/gxingyu/hisearch`. Verification
passed: 209 RL + 91 SFT + 35 shell tests, 532 Python AST checks, Bash syntax,
documentation links, and the tracked-file audit. A fresh environment without
Torch/Ray/vLLM also passed the check-only entry point and 35 lightweight rubric
tests. The release contains 656 tracked files totaling about 6.3 MB; large
resources and credentials are excluded.

The dynamic-rubric path is implemented and verified with offline/mock tests;
real judge calibration and rubric training gains remain to be established.
No policy training or production retrieval/judge service was started for this
publication task. Test daemons and mock servers used temporary synthetic inputs.

See [README.md](README.md), [release status](docs/release_status.md), and
[provenance](docs/PROVENANCE.md) for implementation and validation boundaries.
