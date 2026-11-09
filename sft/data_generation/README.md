# Hybrid trajectory generation

See [the SFT workflow](../README.md) for prerequisites, generation, review,
grouped parquet export, and policy training. The main entry point is
`create_hybrid_data.py`; run it with `--help` for sampling and budget options.

`utils/hybrid_pipeline.py`, `hybrid_tools.py`, `hybrid_prompts.py`, and
`branching.py` implement executed teacher trajectories and shared-prefix
branching. Other helpers cover visible evidence, identity/relationship checks,
conclusions, readiness, and teacher server pooling. Tests use synthetic inputs
and mocked services; they do not establish teacher semantic accuracy.
