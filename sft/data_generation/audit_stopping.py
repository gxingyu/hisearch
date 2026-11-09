#!/usr/bin/env python3
"""Fresh, future-blind stop audit; keep original synthesis outputs immutable."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.hybrid_pipeline import judge_call, parse_action, visible_text
from utils.readiness import COMPARISON_RULES, prefix_payload, valid_citations
from utils.llm import ServerLLM
from utils.server_pool import ServerPool
from utils.answer_grounding import answer_support
from utils.conclusion import check_conclusion, conclusion_payload
from utils.logic import check_short_circuit

PROMPT = """
You audit an assistant immediately before it searches. You receive ONLY past
sources, the question, and its current reasoning. No gold or future answer is
provided. Detect already having derived the COMPLETE requested answer, or
explicitly acknowledging the answer/all requested facts are found, followed by
searching anyway. Confirmation, confidence, aliases unused by the question, or
truncation of an unrelated sentence do not justify another search. One known
counterexample decides 'are BOTH' No; one example decides 'is EITHER' Yes.
Negating certainty is not a counterexample: 'not definitively P' does not assert
not-P, and 'sometimes two species, or four or five' leaves 'just two' unknown.
Keep necessary multi-tool paths: a bridge name with its requested property still
unknown, ambiguous entity, missing part of a list, or a reasonable no-hit followed
by new retrieval is not redundancy. Do not use your own world knowledge. Do not
treat a proposed query, a question restatement, or a hypothesis as an answer.
If reasoning already says the full answer is established, flag it even if it
doesn't repeat the short answer literally. Facts given by the question are source
0; past observations use their displayed source numbers. All quotes must be exact.
Return ONLY JSON {"decision":"redundant"|"continue"|"uncertain",
"reason":string,"reasoning_quote":string|null,"answer":string|null,
"missing_relation":string|null,
"evidence":[{"source":integer,"quote":string}]}.
For continue, specify what fact is still missing. For redundant, quote either
the assistant's complete conclusion/acknowledgment or the sufficient source facts.
""" + COMPARISON_RULES + "\nINPUT:\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--servers", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise ValueError("Use a new audit output directory")
    rows = [json.loads(line) for name in args.input for line in Path(name).read_text().splitlines() if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows)
    pool = ServerPool(args.servers)
    llm = ServerLLM(pool=pool, api_key=os.environ["OPENAI_API_KEY"], num_workers=args.workers, max_retries=2)
    items, malformed, grounding_quarantine = [], set(), []
    for row in rows:
        steps = []
        for index, message in enumerate(row["messages"]):
            if message["role"] != "assistant" or "<tool_call>" not in message["content"]:
                continue
            reasoning, action, _ = parse_action(message["content"])
            if index + 1 >= len(row["messages"]) or row["messages"][index + 1]["role"] != "tool":
                malformed.add(row["id"])
                continue
            observation = json.loads(row["messages"][index + 1]["content"])
            payload = prefix_payload(row["question"], steps, reasoning, visible_text)
            items.append({"id": row["id"], "question_id": row["question_id"],
                          "before_tool": len(steps) + 1, "payload": payload, "action": action,
                          "conclusion_payload": conclusion_payload(row["question"], steps, reasoning)})
            steps.append({"think": reasoning, "action": action, "observation": observation})
        final_message = row["messages"][-1]
        _, final_action, answer = parse_action(final_message["content"])
        assert final_action is None
        grounding = answer_support(row["question"], [visible_text(s["observation"])
            for s in steps if s["observation"].get("status") == "ok"], answer)
        if not grounding["supported"]:
            grounding_quarantine.append({"id": row["id"], "question": row["question"],
                                          "answer": answer, **grounding})
    def run(item):
        logical = check_short_circuit(llm, item["payload"], judge_call)
        if logical["status"] != "continue":
            decision = "redundant" if logical["status"] == "stop" else "uncertain"
            result = {"decision": decision, "reason": logical.get("reason"), "logic": logical}
            return {**item, "result": result, "decision": decision}
        conclusion = check_conclusion(llm, item["conclusion_payload"], judge_call)
        if conclusion["state"] != "incomplete":
            decision = "redundant" if conclusion["state"] == "complete" else "uncertain"
            result = {"decision": decision, "reason": conclusion["reason"],
                      "reasoning_quote": conclusion.get("quote"), "conclusion": conclusion}
            return {**item, "result": result, "decision": decision}
        result = judge_call(llm, PROMPT + json.dumps(item["payload"], ensure_ascii=False))
        decision = result.get("decision")
        if decision == "redundant":
            quote = result.get("reasoning_quote")
            explicit = isinstance(quote, str) and bool(quote.strip()) and quote in item["payload"]["proposed_reasoning"]
            if not explicit and not valid_citations(result, item["payload"]):
                decision = "uncertain"
        elif decision != "continue" or not result.get("missing_relation"):
            decision = "uncertain"
        return {**item, "result": result, "decision": decision, "conclusion": conclusion, "logic": logical}
    results = []
    with (output / "requests_results.jsonl").open("w") as archive, ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run, item) for item in items]):
            result = future.result()
            results.append(result)
            archive.write(json.dumps(result, ensure_ascii=False) + "\n")
            archive.flush()
    rejected = malformed | {r["id"] for r in results if r["decision"] != "continue"} | {
        row["id"] for row in grounding_quarantine}
    kept = [row for row in rows if row["id"] not in rejected]
    quarantined = [row for row in rows if row["id"] in rejected]
    for name, values in [("kept.jsonl", kept), ("quarantine.jsonl", quarantined)]:
        (output / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))
    keywords = re.compile(r"already|answer|confirm|verif|sufficien|found|established|both|truncat|check again|given in|provided in", re.I)
    candidates = [{"id": item["id"], "before_tool": item["before_tool"],
                   "question": item["payload"]["question"], "reasoning": item["payload"]["proposed_reasoning"],
                   "decision": item["decision"], "judge_reason": item["result"].get("reason"),
                   "action": item["action"]}
                  for item in sorted(results, key=lambda r: (r["id"], r["before_tool"]))
                  if keywords.search(item["payload"]["proposed_reasoning"]) or item["decision"] != "continue"]
    (output / "keyword_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    (output / "grounding_quarantine.json").write_text(json.dumps(grounding_quarantine, ensure_ascii=False, indent=2))
    summary = {"input_questions": len({r["question_id"] for r in rows}), "input_trajectories": len(rows),
               "prefixes": len(items), "kept_questions": len({r["question_id"] for r in kept}),
               "kept_trajectories": len(kept), "quarantined_trajectories": len(quarantined),
               "decisions": {d: sum(r["decision"] == d for r in results) for d in ["continue", "redundant", "uncertain"]},
               "keyword_candidates": len(candidates),
               "unsupported_named_answers": len(grounding_quarantine),
               "input_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.input},
               "limitation": "Same 27B, fresh prompt; not independent human ground truth. Inspect keyword/long-path cases."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
