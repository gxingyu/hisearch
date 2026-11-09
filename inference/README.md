# Retrieval environment and inference utilities

The hybrid SFT and RL paths share `retrieval_env/`: tool schemas, dense HTTP
client, original corpus lookup, candidate sessions, scoped shell execution,
and preflight checks. `parallel_search/` implements full-corpus sharding,
execution, daemon/client communication, caching, and process cleanup.

The legacy `run.py`, `agent.py`, and `run_inference.sh` are shell-only utilities
inherited from GrepSeek. They do not provide the hybrid policy evaluation
protocol. Use the validation-only training agent described in
[rl/README.md](../rl/README.md) to evaluate hybrid models.

The Python `grepseek` package lives in `rl/`. From the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/rl:$PWD/verl:${PYTHONPATH:-}"
python -m inference.retrieval_env.preflight --health-only
python -m unittest discover -s inference/parallel_search/tests -v
```

The first command checks an already-running dense endpoint. Unit tests create
temporary synthetic corpora and local processes; they do not search the full
Wikipedia snapshot or start policy training.
