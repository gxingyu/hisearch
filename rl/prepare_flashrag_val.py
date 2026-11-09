#!/usr/bin/env python3
"""Build the mixed FlashRAG validation set used by GrepSeek RL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DATASET_FILES = {
    "bamboogle": "bamboogle/test.jsonl",
    "2wikimultihopqa": "2wikimultihopqa/dev.jsonl",
    "hotpotqa": "hotpotqa/dev.jsonl",
    "musique": "musique/dev.jsonl",
    "nq": "nq/test.jsonl",
    "popqa": "popqa/test.jsonl",
    "triviaqa": "triviaqa/test.jsonl",
}
SAMPLE_SIZES = {"bamboogle": None, **{name: 500 for name in DATASET_FILES if name != "bamboogle"}}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_record(record: dict, source: str) -> dict:
    record_id = record.get("id") or record.get("qid")
    question = record.get("question") or record.get("query")
    answers = record.get("golden_answers")
    if record_id is None or question is None or not isinstance(answers, list) or not answers:
        raise ValueError(f"Invalid QA record in {source}: {record!r}")
    return {
        "id": f"{source}:{record_id}",
        "question": str(question),
        "golden_answers": [str(answer) for answer in answers],
        "source": source,
        "source_id": str(record_id),
    }


def build_validation_set(dataset_root: Path, output_path: Path, seed: int) -> dict:
    rng = random.Random(seed)
    selected: list[dict] = []
    summary: dict[str, dict[str, int]] = {}

    for source, relative_path in DATASET_FILES.items():
        source_path = dataset_root / relative_path
        records = _read_jsonl(source_path)
        sample_size = SAMPLE_SIZES[source]
        if sample_size is None:
            chosen = records
        else:
            if len(records) < sample_size:
                raise ValueError(f"{source} has {len(records)} records; need {sample_size}")
            chosen = rng.sample(records, sample_size)
        normalized = [_normalize_record(record, source) for record in chosen]
        selected.extend(normalized)
        summary[source] = {
            "available": len(records),
            "selected": len(normalized),
        }

    ids = [record["id"] for record in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs found in mixed validation set")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "seed": seed,
        "total": len(selected),
        "datasets": summary,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/flashrag_datasets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rl/flashrag_7bench/val.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_validation_set(args.dataset_root, args.output, args.seed)
    summary_path = args.output.with_name("build_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
