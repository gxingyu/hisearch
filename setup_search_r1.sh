#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
uv venv --python 3.10 --allow-existing venv
uv pip install --python venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python venv/bin/python -r retrieval/requirements.txt
venv/bin/python retrieval/prepare.py "$@"
