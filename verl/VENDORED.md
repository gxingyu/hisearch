# Vendored runtime provenance

The runtime originates from [volcengine/verl](https://github.com/volcengine/verl)
under Apache-2.0, as vendored by GrepSeek. HiSearch exports the Python runtime
from the current research source snapshot, including existing uncommitted local
fixes; it does not substitute a stock pip version or claim an unknown upstream
commit matches those changes.

The source snapshot is based on GrepSeek commit
`1f6ea58` plus local hybrid retrieval, SFT, reward, and training changes.
See [the project provenance](../docs/PROVENANCE.md) for the main modifications.

Local changes include Qwen3.5 and checkpoint compatibility, rollout/parser and
reward handling, dynamic group sampling, the driver's rubric lifecycle,
checkpointed rubric state, and optional historical reward controls. The rubric
recipe disables the optional stop, gate, route-efficiency, and length controls.

The `verl/` Python package, configuration files, and version resource are retained.
Upstream development documentation, examples, recipes, tests, CI, containers,
editor configuration, and example shell launchers are omitted. Runtime modules
are kept together to preserve dynamic imports and configuration compatibility.
The original copyright notices, `LICENSE`, and `Notice.txt` are preserved.
