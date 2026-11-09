from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from inference.retrieval_env.corpus_store import CorpusStore
from inference.retrieval_env.dense_client import DenseClient, DenseServiceError


def check_prompts(tokenizer, system_prompt: str, data_files: list[str], max_length: int) -> dict[str, int]:
    from grepseek.data.qa import load_qa_examples
    from grepseek.prompting import build_initial_messages

    maxima = {}
    for filename in data_files:
        examples = [example for example in load_qa_examples(filename) if example.golden_answers]
        if not examples:
            raise ValueError(f"No answerable QA examples in {filename}")
        maximum = 0
        for start in range(0, len(examples), 128):
            batch = examples[start : start + 128]
            conversations = [
                build_initial_messages(query=example.query, system_prompt=system_prompt, retrieval_instruction="")
                for example in batch
            ]
            encoded = tokenizer.apply_chat_template(
                conversations, tools=[], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            elif isinstance(encoded, dict):
                encoded = encoded["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            for example, token_ids in zip(batch, encoded, strict=True):
                length = len(token_ids)
                if length > max_length:
                    raise ValueError(
                        f"Question {example.qid} in {filename} needs {length} prompt tokens; "
                        f"budget remains {max_length}"
                    )
                maximum = max(maximum, length)
        maxima[filename] = maximum
    return maxima


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Verify hybrid retrieval before starting GSPO")
    parser.add_argument("--url", default=os.environ.get("GREPSEEK_DENSE_URL", "http://127.0.0.1:8000/retrieve"))
    parser.add_argument(
        "--corpus-root", default=os.environ.get("GREPSEEK_CORPUS_ROOT", str(root / "data/wiki_18_corpus"))
    )
    parser.add_argument("--data-root", default=os.environ.get("GREPSEEK_DENSE_DATA_ROOT", str(root / "data/search_r1")))
    parser.add_argument("--model", default=os.environ.get("GREPSEEK_MODEL_PATH"))
    parser.add_argument("--train", default=os.environ.get("GREPSEEK_TRAIN_FILES"))
    parser.add_argument("--val", default=os.environ.get("GREPSEEK_VAL_FILES"))
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--health-only", action="store_true")
    modes.add_argument("--local-only", action="store_true")
    modes.add_argument("--service-only", action="store_true")
    args = parser.parse_args()
    client = DenseClient(args.url, args.timeout)
    if args.health_only:
        print(json.dumps(client.health()), flush=True)
        return
    store = CorpusStore(Path(args.corpus_root) / "wiki_corpus.jsonl", args.data_root)
    report = {"corpus_version": store.version, "dense_topk": 3}
    if not args.service_only:
        from grepseek.trainer.verl_integration.hybrid_dataset import build_hybrid_system_prompt
        from transformers import AutoTokenizer

        from verl.tools.utils.tool_registry import initialize_tools_from_config

        if not args.model or not (Path(args.model) / "config.json").is_file():
            raise ValueError("The same local SFT checkpoint used by run_gspo.sh is required")
        if not args.train or not args.val:
            raise ValueError("Both training and validation QA files are required")
        store.verify_snapshot()
        config = {
            "system_prompt_path": root / "rl/prompts/system_prompt_hybrid.txt",
            "retrieval_instruction_path": root / "rl/prompts/output_format_instruction.txt",
            "tool_spec_template_path": root / "rl/prompts/tool_spec_template_hybrid.txt",
        }
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
        report["max_prompt_tokens"] = check_prompts(
            tokenizer, build_hybrid_system_prompt(config), [args.train, args.val], args.max_prompt_length
        )
        os.environ["GREPSEEK_ROOT"] = str(root)
        os.environ["VERL_TOOL_CORPUS_ROOT"] = str(Path(args.corpus_root).resolve())
        os.environ["GREPSEEK_DENSE_DATA_ROOT"] = str(Path(args.data_root).resolve())
        previous_engine = os.environ.get("GREPSEEK_SEARCH_ENGINE")
        os.environ["GREPSEEK_SEARCH_ENGINE"] = "none"
        try:
            tools = initialize_tools_from_config(root / "rl/grepseek/trainer/config/tools/search_tools_hybrid.yaml")
        finally:
            if previous_engine is None:
                os.environ.pop("GREPSEEK_SEARCH_ENGINE", None)
            else:
                os.environ["GREPSEEK_SEARCH_ENGINE"] = previous_engine
        report["tools"] = [tool.name for tool in tools]
    if not args.local_only:
        health = client.health()
        if health["corpus_sha256"] != store.manifest["sha256"] or health.get("rows") != store.manifest["count"]:
            raise ValueError("Dense service and shell corpus snapshots differ")
        results = client.retrieve("Who is Marie Curie's spouse?")
        if len(results) != 3:
            raise ValueError("Dense service smoke test did not return exactly 3 passages")
        for result in results:
            _, original = store.read(result["document"]["id"])
            if original.get("contents") != result["document"]["contents"]:
                raise ValueError("Dense response differs from the original shell corpus")
        report["health"] = health
        report["retrieved_passage_ids"] = [str(result["document"]["id"]) for result in results]
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (DenseServiceError, ValueError, OSError) as error:
        print(f"Hybrid preflight failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
