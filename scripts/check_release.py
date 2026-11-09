#!/usr/bin/env python3
"""Audit tracked source files without printing matched credential values."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_ROOTS = {
    "data", "models", "checkpoints", "outputs", "output", "wandb",
    ".cache", ".venv", "venv",
}
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".index", ".npy", ".npz",
    ".parquet", ".arrow", ".jsonl", ".log", ".zip", ".gz", ".tar", ".pem", ".key",
}
SECRET_PATTERNS = {
    "API credential pattern": re.compile(
        r"(?:wandb_v1_[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{15,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential in URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-file-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
    )
    paths = [Path(p.decode()) for p in result.stdout.split(b"\0") if p]
    if not paths:
        parser.error("No tracked files; stage the intended release before checking")
    errors = []
    total = 0
    largest = (0, "")
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink():
            errors.append(f"{relative}: symlinks are not allowed in this source release")
            continue
        if relative.parts[0] in BLOCKED_ROOTS or "__pycache__" in relative.parts:
            errors.append(f"{relative}: runtime artifact directory")
        if relative.suffix in BLOCKED_SUFFIXES:
            errors.append(f"{relative}: binary/data/artifact extension")
        if relative.name.startswith(".env") and relative.name != ".env.example":
            errors.append(f"{relative}: private environment file")
        payload = path.read_bytes()
        total += len(payload)
        largest = max(largest, (len(payload), str(relative)))
        if len(payload) > args.max_file_bytes:
            errors.append(f"{relative}: exceeds {args.max_file_bytes} bytes")
        try:
            contents = payload.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{relative}: non-UTF-8 file")
            continue
        if "\0" in contents:
            errors.append(f"{relative}: binary content")
        for number, line in enumerate(contents.splitlines(), 1):
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    errors.append(f"{relative}:{number}: {label} (value suppressed)")
    print(f"Checked {len(paths)} tracked files; {total:,} bytes; "
          f"largest {largest[1]} ({largest[0]:,} bytes).")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Source release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
