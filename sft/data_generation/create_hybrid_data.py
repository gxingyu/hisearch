"""Sample and audit multiple executed retrieval strategies per training question."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rl")]

from transformers import AutoTokenizer

from utils.hybrid_pipeline import STRATEGIES, build_system_prompt, generate_trajectory, prepare_reference, sft_record
from utils.hybrid_tools import HybridToolEnvironment
from utils.llm import SamplingParams, ServerLLM
from utils.server_pool import ServerPool
from utils.load_hotpotqa import load_hotpotqa
from utils.load_nq import load_nq
from utils.branching import run_question


def read_records(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def generation_fingerprint():
    digest = hashlib.sha256()
    for relative_path in (
        "sft/data_generation/create_hybrid_data.py",
        "sft/data_generation/utils/hybrid_pipeline.py",
        "sft/data_generation/utils/hybrid_prompts.py",
        "sft/data_generation/utils/hybrid_tools.py",
        "sft/data_generation/utils/branching.py",
        "sft/data_generation/utils/readiness.py",
        "sft/data_generation/utils/answer_grounding.py",
        "sft/data_generation/utils/conclusion.py",
        "sft/data_generation/utils/logic.py",
        "sft/data_generation/utils/birth_order.py",
        "rl/prompts/system_prompt_hybrid.txt",
        "rl/prompts/tool_spec_template_hybrid.txt",
        "inference/retrieval_env/schemas.py",
        "inference/retrieval_env/observations.py",
        "inference/retrieval_env/candidate_executor.py",
    ):
        digest.update(relative_path.encode())
        digest.update((ROOT / relative_path).read_bytes())
    return digest.hexdigest()


def summarize(records):
    by_strategy = defaultdict(list)
    for record in records:
        by_strategy[record["strategy"]].append(record)
    strategies = {}
    for strategy, group in by_strategy.items():
        calls = [step for record in group for step in record.get("steps", [])]
        strategies[strategy] = {
            "sampled": len(group), "kept": sum(record["success"] for record in group),
            "answer_em": sum(record.get("em", False) for record in group),
            "mixed_sampled": sum(record.get("mixed", False) for record in group),
            "mixed_kept": sum(record["success"] and record.get("mixed", False) for record in group),
            "mean_calls": len(calls) / len(group),
            "tool_seconds": sum(step["metrics"]["elapsed_seconds"] for step in calls),
            "abort_reasons": dict(Counter(record["abort_reason"] for record in group if not record["success"])),
            "paths_kept": dict(Counter(record["path"] for record in group if record["success"])),
        }
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(), "population": "all sampled trajectories",
        "questions": len({record["question_id"] for record in records}), "sampled": len(records),
        "kept": sum(record["success"] for record in records), "strategies": strategies,
        "selected": sum(record["success"] and record.get("selected", True) for record in records),
        "executed_tool_requests": sum(record.get("executed_calls", len(record.get("steps", []))) for record in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--servers", help="JSON teacher server pool; replaces --base-url")
    parser.add_argument("--tokenizer", required=True, help="Actor tokenizer used for observation and sequence budgets")
    parser.add_argument("--input", help="Local QA JSONL with id, question, golden_answers; training data only")
    parser.add_argument("--dataset", choices=("hotpotqa", "nq"), default="hotpotqa")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train", help="Source split; SFT synthesis must use train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--samples-per-strategy", type=int, default=1)
    parser.add_argument("--corpus-dir", default=str(ROOT / "data/wiki_18_corpus"))
    parser.add_argument("--data-root", default=str(ROOT / "data/search_r1"))
    parser.add_argument("--dense-url", default=os.environ.get("GREPSEEK_DENSE_URL", "http://127.0.0.1:8000/retrieve"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-tool-calls", type=int, default=5)
    parser.add_argument("--tool-max-tokens", type=int, default=1024)
    parser.add_argument("--tool-timeout", type=float, default=60)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--parallel-examples", type=int, default=16)
    parser.add_argument("--llm-workers", type=int, default=32)
    parser.add_argument("--branching", action="store_true")
    parser.add_argument("--exploration-calls", type=int, default=40)
    parser.add_argument("--max-states", type=int, default=2)
    parser.add_argument("--backward-reference", action="store_true", help="Build internal references once per question; never show to policy")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.servers and (not args.base_url or not args.model):
        parser.error("--base-url/--model or --servers are required")
    if min(args.n, args.samples_per_strategy, args.max_tool_calls, args.tool_max_tokens, args.max_sequence_tokens, args.parallel_examples, args.llm_workers) < 1 or args.start < 0:
        parser.error("Budgets/counts must be positive and --start nonnegative")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set OPENAI_API_KEY in the process environment")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    environment = HybridToolEnvironment.from_paths(
        args.corpus_dir, args.data_root, args.dense_url, tokenizer, args.tool_max_tokens, args.tool_timeout,
    )
    pool = ServerPool(args.servers) if args.servers else None
    if pool:
        pool.start_reloader()
        llm = ServerLLM(pool=pool, model=args.model, api_key=api_key, max_retries=2, num_workers=args.llm_workers)
    else:
        llm = ServerLLM(base_url=args.base_url, model=args.model, api_key=api_key, max_retries=2, num_workers=args.llm_workers)
    if llm.client is not None:
        llm.client.timeout = 180
        llm.client.max_retries = 0
    if args.preflight_only:
        response = llm.chat(
            [{"role": "user", "content": "Reply with OK."}],
            SamplingParams(max_tokens=32, extra_body={"thinking": {"type": "disabled"}}),
        )
        if not response.text.strip():
            raise ValueError("Teacher returned empty content in preflight")
        print(json.dumps({"status": "ok", "model": args.model, "corpus_version": environment.store.version}))
        return
    if args.input:
        examples = read_records(Path(args.input))[args.start:args.start + args.n]
    else:
        loader = load_hotpotqa if args.dataset == "hotpotqa" else load_nq
        if args.split != "train":
            parser.error("SFT synthesis requires the train split")
        examples = loader(split=args.split, limit=args.start + args.n)[args.start:args.start + args.n]
    if not examples or any(not example.get("id") or not example.get("question") or not example.get("golden_answers") for example in examples):
        parser.error("Need training examples with id, question, and nonempty golden_answers")
    if len({str(example["id"]) for example in examples}) != len(examples):
        parser.error("Input question ids must be unique")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "traces.jsonl"
    reference_path = output / "references.jsonl"
    config = {**vars(args), "corpus_version": environment.store.version}
    config["generator_sha256"] = generation_fingerprint()
    config["questions_sha256"] = hashlib.sha256(json.dumps(examples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    config_path = output / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if {key: value for key, value in previous.items() if key != "resume"} != {key: value for key, value in config.items() if key != "resume"}:
            parser.error("Resume configuration differs from the saved run")
    elif any(output.iterdir()):
        parser.error("Output directory is nonempty; use --resume or a new directory")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    question_directory = output / "questions"
    if args.resume and args.branching and question_directory.exists():
        completed_results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(question_directory.glob("*.json"))]
        records = []
        for result in completed_results:
            for record in result["records"]:
                record["question_complete"] = True
                records.append(record)
        for filename, key in (("states.jsonl", "states"), ("counterfactuals.jsonl", "counterfactuals"), ("compression.jsonl", "compression")):
            (output / filename).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for result in completed_results for item in result[key]), encoding="utf-8")
    else:
        records = read_records(raw_path) if args.resume else []
    if args.resume:
        with raw_path.open("w", encoding="utf-8") as raw_file:
            for record in records:
                raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    completed = {record["trajectory_id"] for record in records}
    if args.resume:
        sync_outputs(output, records)
    references = {record["question_id"]: record for record in read_records(reference_path)} if args.resume else {}
    system_prompt = build_system_prompt(ROOT)
    if args.branching:
        if args.samples_per_strategy != 1 or args.backward_reference:
            parser.error("Branching uses one baseline per strategy and no backward references")
        completed_questions = {record["question_id"] for record in records if record.get("question_complete")}
        def run_example(example):
            return run_question(llm, environment, example, system_prompt, args.strategies,
                args.max_tool_calls, args.temperature, args.max_sequence_tokens,
                args.exploration_calls, args.max_states)
        pending = [example for example in examples if str(example["id"]) not in completed_questions]
        question_directory.mkdir(exist_ok=True)
        with raw_path.open("a", encoding="utf-8") as raw_file, (output / "states.jsonl").open("a", encoding="utf-8") as state_file, (output / "counterfactuals.jsonl").open("a", encoding="utf-8") as comparison_file, (output / "compression.jsonl").open("a", encoding="utf-8") as compression_file, ThreadPoolExecutor(max_workers=args.parallel_examples) as pool:
            futures = {pool.submit(run_example, example): example for example in pending}
            for future in as_completed(futures):
                result = future.result()
                question_path = question_directory / (hashlib.sha256(result["question_id"].encode()).hexdigest() + ".json")
                temporary_path = question_path.with_suffix(".tmp")
                temporary_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                temporary_path.replace(question_path)
                for record in result["records"]:
                    record["question_complete"] = True
                    raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                for state in result["states"]:
                    state_file.write(json.dumps(state, ensure_ascii=False) + "\n")
                for comparison in result["counterfactuals"]:
                    comparison_file.write(json.dumps(comparison, ensure_ascii=False) + "\n")
                for compression in result["compression"]:
                    compression_file.write(json.dumps(compression, ensure_ascii=False) + "\n")
                raw_file.flush()
                state_file.flush()
                comparison_file.flush()
                compression_file.flush()
                sync_outputs(output, records)
                print(f"question={result['question_id']} sampled={len(result['records'])} selected={len(result['selected_trajectories'])} calls={result['executed_calls']}", flush=True)
        return
    with raw_path.open("a", encoding="utf-8") as raw_file:
        for example in examples:
            question_id = str(example["id"])
            if args.backward_reference and question_id not in references:
                reference = prepare_reference(llm, example, args.corpus_dir, max_calls=args.max_tool_calls)
                references[question_id] = {"question_id": question_id, **reference}
                with reference_path.open("a", encoding="utf-8") as reference_file:
                    reference_file.write(json.dumps(references[question_id], ensure_ascii=False) + "\n")
            jobs = []
            for strategy in args.strategies:
                for sample_index in range(args.samples_per_strategy):
                    trajectory_id = f"{question_id}::{strategy}::{sample_index}"
                    if trajectory_id not in completed:
                        jobs.append((trajectory_id, strategy, sample_index))
            def run_job(job):
                trajectory_id, strategy, sample_index = job
                started = time.monotonic()
                record = generate_trajectory(
                    llm, environment, example, system_prompt, trajectory_id, strategy,
                    args.max_tool_calls, args.temperature, max_sequence_tokens=args.max_sequence_tokens,
                    reference=references.get(question_id), stop_when_ready=True,
                )
                record["elapsed_seconds"] = time.monotonic() - started
                record["completed_at"] = datetime.now(timezone.utc).isoformat()
                return record
            with ThreadPoolExecutor(max_workers=args.parallel_examples) as pool:
                futures = [pool.submit(run_job, job) for job in jobs]
                for future in as_completed(futures):
                    record = future.result()
                    raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raw_file.flush()
                    records.append(record)
                    completed.add(record["trajectory_id"])
                    sync_outputs(output, records)
                    print(f"{record['trajectory_id']}: kept={record['success']} path={record['path']} reason={record['abort_reason']}", flush=True)


def sync_outputs(output, records):
    with (output / "sft.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            if record.get("selected") is False:
                continue
            exported = sft_record(record)
            if exported:
                stream.write(json.dumps(exported, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
