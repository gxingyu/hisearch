"""Executed forward sampling and conservative evidence filtering."""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping

from inference.retrieval_env.schemas import tool_schema, tool_schemas, validate_parameters
from .hybrid_prompts import ABLATION_PROMPT, ANSWER_EQUIVALENCE_PROMPT, POLICY_INSTRUCTION, PROVENANCE_PROMPT, REVIEW_PROMPT
from .llm import SamplingParams
from .pipeline import decompose, qa_exact_match, qa_f1_score, run_backward
from .readiness import audit_prefixes, check_ready, prefix_payload
from .answer_grounding import answer_support
from .conclusion import check_conclusion, conclusion_payload
from .logic import check_short_circuit


STRATEGIES = ("free", "dense_first", "shell_first", "shell_only", "dense_only")


def _normalized_phrase(text):
    """Normalize harmless punctuation/case differences for provenance checks."""
    text = re.sub(r"\b[Uu]\.[Ss]\.(?:[Aa]\.)?|\bUSA\b|\bUS\b", "United States", text)
    text = re.sub(r"\b[Uu]\.[Kk]\.|\bUK\b", "United Kingdom", text)
    text = re.sub(r"['’]", "", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _phrase_supported(phrase, text):
    normalized_phrase = _normalized_phrase(phrase)
    normalized_text = _normalized_phrase(text)
    return bool(normalized_phrase) and normalized_phrase in normalized_text


def json_request_body(llm, seed=None):
    model = getattr(llm, "model", "")
    body = {"response_format": {"type": "json_object"}}
    if isinstance(model, str) and "qwen" in model.lower():
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["thinking"] = {"type": "disabled"}
    if seed is not None:
        body["seed"] = seed
    return body


def build_system_prompt(root):
    prompt = (root / "rl/prompts/system_prompt_hybrid.txt").read_text(encoding="utf-8").strip()
    specification = (root / "rl/prompts/tool_spec_template_hybrid.txt").read_text(encoding="utf-8").format(
        tool_spec=json.dumps(tool_schemas(), ensure_ascii=False),
    ).strip()
    return "\n\n".join((prompt, specification, POLICY_INSTRUCTION.strip()))


def parse_action(text):
    calls = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    answers = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL))
    if len(calls) + len(answers) != 1:
        raise ValueError("Expected exactly one tool_call or answer")
    block = (calls or answers)[0]
    reasoning = re.sub(r"</?think>", "", text[:block.start()]).strip()
    if not reasoning or text[block.end():].strip():
        raise ValueError("Expected reasoning before the action and no trailing text")
    if answers:
        answer = block.group(1).strip()
        if not answer:
            raise ValueError("Empty answer")
        return reasoning, None, answer
    action = json.loads(block.group(1))
    if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
        raise ValueError("Tool call requires name and arguments only")
    if action["name"] not in ("shell", "dense_search"):
        raise ValueError("Unknown tool")
    validate_parameters(action["arguments"], tool_schema(action["name"])["function"]["parameters"])
    return reasoning, action, None


def parse_json(text):
    cleaned = text.strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence) and cleaned.endswith(fence):
        cleaned = cleaned.split("\n", 1)[1].rsplit(fence, 1)[0]
    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("Judge must return a JSON object")
    return result


def visible_text(observation):
    if observation.get("tool") == "dense_search":
        return "\n".join(document.get("contents", "") for document in observation.get("documents", []))
    stdout = observation.get("stdout", "")
    decoded = []
    for line in stdout.splitlines():
        try:
            document = json.loads(line)
        except ValueError:
            decoded.append(line)
            continue
        if isinstance(document, dict) and isinstance(document.get("contents"), str):
            decoded.append(document["contents"])
        else:
            decoded.append(line)
    return "\n".join(decoded) if decoded else stdout


def sequence_length(tokenizer, messages):
    encoded = tokenizer.apply_chat_template(
        messages, tools=tool_schemas(), tokenize=True, add_generation_prompt=False,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return len(token_ids)


def unseen_gold(text, gold_answers, frontier):
    return any(
        len(gold.strip()) >= 3 and gold.strip().lower() not in ("yes", "true", "false", "no")
        and gold.strip().lower() in text.lower() and gold.strip().lower() not in frontier.lower()
        for gold in gold_answers
    )


def unseen_numbers(text, frontier):
    return any(number not in frontier for number in re.findall(r"\b\d{3,4}\b", text))


def propose_step(llm, teacher_messages, record, frontier, required_kind, turn,
                 max_calls, temperature, max_response_tokens, seed, retries,
                 check_numbers, excluded_action, allowed_tool=None):
    prompt = copy.deepcopy(teacher_messages)
    if retries:
        reminder = (
            "Before replying: give one short reasoning sentence grounded in the visible history. "
            "If the answer is already supported, answer now, including when a tool path was suggested. "
            "Otherwise name the missing relation. Never introduce remembered facts or aliases. "
            "Use only entities explicitly present in the question or previous visible observations. "
            "Every shell command must explicitly read corpus.jsonl. Return the specified JSON object. "
        )
        if required_kind:
            reminder += f"If retrieval is still needed, next action category: {required_kind}. "
        prompt.append({"role": "user", "content": reminder})
    failure = "invalid_action"
    for attempt in range(retries + 1):
        response = llm.chat(prompt, SamplingParams(
            temperature=temperature, max_tokens=max_response_tokens,
            extra_body=json_request_body(llm, None if seed is None else seed + turn + attempt * 10000),
        ))
        raw = response.text or ""
        record["model_responses"].append(raw)
        try:
            proposed = parse_json(raw)
            reasoning = proposed.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("Missing reasoning")
            if reasoning.strip().lower() in ("brief evidence-grounded prose", "evidence-grounded prose", "...",
                                             "<your grounded reasoning>", "<your grounded conclusion>"):
                raise ValueError("Placeholder reasoning")
            action, answer = proposed.get("action"), proposed.get("answer")
            if (action is None) == (answer is None):
                raise ValueError("Exactly one action or answer is required")
            if action is not None:
                reasoning, action, _ = parse_action(reasoning + "\n<tool_call>" + json.dumps(action, ensure_ascii=False) + "</tool_call>")
            elif not isinstance(answer, str) or not answer.strip():
                raise ValueError("Empty answer")
            query = action["arguments"].get("command", action["arguments"].get("query", "")) if action else ""
            kind = "stop" if action is None else "shell_candidates" if action["name"] == "shell" and action["arguments"].get("scope", {}).get("type") == "candidates" else action["name"]
            failure = ""
            if action is not None and unseen_gold(reasoning, record["gold_answers"], frontier):
                failure = "unseen_gold_in_reasoning"
            elif unseen_gold(query, record["gold_answers"], frontier):
                failure = "unseen_gold_in_query"
            elif action is not None and check_numbers and unseen_numbers(reasoning + "\n" + query, frontier):
                failure = "unseen_number"
            elif required_kind and kind != required_kind and action is not None:
                failure = "sampling_strategy_violation"
            elif allowed_tool and action is not None and kind != allowed_tool:
                failure = "sampling_strategy_violation"
            elif excluded_action is not None and action is not None and action == excluded_action:
                failure = "compression_repeated_deleted_action"
            elif turn == max_calls and action is not None:
                failure = "tool_budget_exhausted"
            elif retries and action is not None and action["name"] == "shell" and "corpus.jsonl" not in query:
                failure = "missing_corpus_file"
            if not failure:
                return reasoning, action, answer, None
        except (ValueError, TypeError) as error:
            failure = f"invalid_action: {error}"
        record.setdefault("proposal_rejections", []).append({"turn": turn + 1, "attempt": attempt, "reason": failure})
        if attempt < retries:
            prompt.append({"role": "user", "content":
                "The previous proposal violated the retrieval protocol. Generate a fresh proposal from "
                "the same visible history. Do not state or search a remembered answer/alias/date. "
                "Answer now if already supported; otherwise follow the retrieval category and use corpus.jsonl in shell commands. "
                + (f"Retrieval category: {required_kind}." if required_kind else "")})
    return None, None, None, failure


def leakage_failure(record):
    frontier = record["question"]
    for step in record["steps"]:
        if unseen_gold(step["think"], record["gold_answers"], frontier):
            return "unseen_gold_in_reasoning"
        arguments = step["action"]["arguments"]
        if unseen_gold(arguments.get("command", arguments.get("query", "")), record["gold_answers"], frontier):
            return "unseen_gold_in_query"
        frontier += "\n" + visible_text(step["observation"])
    # A computed/derived final need not occur verbatim in a source. Only the
    # audited protocol may use this exemption; legacy/unchecked records cannot.
    audited_final = (record.get("stop_protocol") in ("prefix_v1", "prefix_v2")
                     and (record.get("provenance_check") or {}).get("verdict") == "PASS"
                     and (record.get("quality_check") or {}).get("verdict") == "PASS"
                     and record.get("stop_audit_passed") is True)
    if record.get("final") and not audited_final and unseen_gold(record["final"]["think"], record["gold_answers"], frontier):
        return "unseen_gold_in_reasoning"
    return None


def judge_call(llm, prompt, max_tokens=2048):
    last_raw = ""
    for attempt in range(2):
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            SamplingParams(
                temperature=0, max_tokens=max_tokens,
                extra_body=json_request_body(llm),
            ),
        )
        last_raw = response.text or ""
        try:
            return {**parse_json(last_raw), "raw_response": last_raw}
        except ValueError:
            continue
    return {"verdict": "FAIL", "reason": "judge_parse_failed", "raw_response": last_raw}


def audit_provenance(llm, record):
    turns = []
    sources = {0: record["question"]}
    for turn, step in enumerate(record["steps"], 1):
        arguments = step["action"]["arguments"]
        turns.append({"turn": turn, "reasoning": step["think"],
                      "query": arguments.get("command", arguments.get("query", ""))})
        sources[turn] = visible_text(step["observation"]) if step["observation"].get("status") == "ok" else ""
    turns.append({"turn": len(turns) + 1, "reasoning": record["final"]["think"], "query": ""})
    review = judge_call(llm, PROVENANCE_PROMPT + json.dumps({"question": record["question"],
        "turns": turns, "visible_sources": sources}, ensure_ascii=False), max_tokens=4096)
    if review.get("verdict") != "PASS" or review.get("complete") is not True or not isinstance(review.get("mentions"), list):
        return review, "provenance_review_failed"
    for mention in review["mentions"]:
        if not isinstance(mention, dict):
            return review, "invalid_provenance_mention"
        turn, source_turn = mention.get("turn"), mention.get("source_turn")
        phrase, quote = mention.get("text"), mention.get("quote")
        if type(turn) is not int or type(source_turn) is not int or not 1 <= turn <= len(turns) or not 0 <= source_turn < turn:
            return review, "invalid_provenance_chronology"
        target = turns[turn - 1]["reasoning"] + "\n" + turns[turn - 1]["query"]
        if not isinstance(phrase, str) or not phrase or phrase not in target:
            return review, "invalid_provenance_target"
        ordinary_relation = _normalized_phrase(phrase) in {
            "telephone", "phone", "birthplace", "born", "release date", "released", "country",
            "nationality", "author", "written", "director", "directed", "movie", "film", "singer",
            "company", "manufacturer", "state", "city", "town", "year", "date"}
        final_derivation = turn == len(turns) and isinstance(quote, str) and answer_support(
            quote, [], phrase)["supported"]
        if (not isinstance(quote, str) or not quote or quote not in sources[source_turn]
                or not (_phrase_supported(phrase, quote) or ordinary_relation or final_derivation)):
            return review, "invalid_provenance_source"
        mention["source_start"] = sources[source_turn].index(quote)
        mention["source_end"] = mention["source_start"] + len(quote)
    return review, None


def answer_equivalence(llm, record):
    observations = [{"turn": turn, "contents": visible_text(step["observation"])}
                    for turn, step in enumerate(record["steps"], 1) if step["observation"].get("status") == "ok"]
    result = judge_call(llm, ANSWER_EQUIVALENCE_PROMPT + json.dumps({
        "question": record["question"], "prediction": record["final"]["answer"],
        "gold_answers": record["gold_answers"], "observations": observations}, ensure_ascii=False))
    evidence = result.get("evidence")
    if result.get("equivalent") is not True or not isinstance(evidence, list) or not evidence:
        return result, False
    sources = {0: record["question"], **{item["turn"]: item["contents"] for item in observations}}
    for entry in evidence:
        if not isinstance(entry, dict) or type(entry.get("turn")) is not int or not isinstance(entry.get("quote"), str) or not entry["quote"]:
            return result, False
        if entry["quote"] not in sources.get(entry["turn"], ""):
            return result, False
    predicted_numbers = set(re.findall(r"\b\d+\b", record["final"]["answer"]))
    if predicted_numbers and not any(set(re.findall(r"\b\d+\b", gold)) <= predicted_numbers
            for gold in record["gold_answers"] if re.search(r"\d", gold)):
        return result, False
    return result, True


def validate_review(review, record):
    if review.get("verdict") != "PASS" or any(
        review.get(key) is not True for key in ("answer_correct", "grounded", "complete_evidence")
    ):
        return "review_failed"
    turns = review.get("turns")
    if not isinstance(turns, list) or len(turns) != len(record["steps"]):
        return "incomplete_turn_review"
    turn_ids = [entry.get("turn") for entry in turns if isinstance(entry, dict)]
    if any(type(turn) is not int for turn in turn_ids) or sorted(turn_ids) != list(range(1, len(record["steps"]) + 1)):
        return "invalid_review_turns"
    if any(entry.get("contribution") not in ("evidence", "disambiguation", "recovery") for entry in turns):
        return "redundant_or_unreviewed_turn"
    claims = review.get("claims")
    if not isinstance(claims, list) or not claims:
        return "missing_evidence_graph"
    seen = set()
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str) or not claim.get("claim"):
            return "invalid_claim"
        if claim["id"] in seen or not isinstance(claim.get("depends_on"), list):
            return "invalid_claim_dependencies"
        if any(not isinstance(dependency, str) or dependency not in seen for dependency in claim["depends_on"]):
            return "invalid_claim_dependencies"
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            return "missing_claim_evidence"
        for entry in evidence:
            if not isinstance(entry, dict):
                return "invalid_quote"
            turn, quote = entry.get("turn"), entry.get("quote")
            if type(turn) is not int or not 0 <= turn <= len(record["steps"]):
                return "invalid_quote_turn"
            if not isinstance(quote, str) or not quote.strip():
                return "invalid_quote"
            if turn == 0:
                if quote not in record.get("question", ""):
                    return "quote_not_in_visible_evidence"
                continue
            observation = record["steps"][turn - 1]["observation"]
            if observation.get("status") != "ok" or quote not in visible_text(observation):
                return "quote_not_in_visible_evidence"
        seen.add(claim["id"])
    return None


def prepare_reference(llm, example, corpus_dir, max_iterations=3, max_calls=5):
    if example.get("type") == "comparison" or any(
        answer.strip().lower() in ("yes", "no", "true", "false")
        for answer in example.get("golden_answers", [])
    ):
        return {"status": "not_applicable", "reason": "linear_backward_does_not_cover_branch_comparison"}
    answers = example.get("golden_answers") or []
    if not answers:
        return {"status": "not_applicable", "reason": "missing_gold"}
    decomposition = decompose(llm, example["question"], answers[0])
    if not decomposition["chain"] or len(decomposition["chain"]) > max_calls:
        return {"status": "failed", "decomposition": decomposition}
    backward = run_backward(
        llm, example["question"], answers[0], decomposition["chain"],
        corpus_dir=corpus_dir, max_iterations=max_iterations, tool_max_tokens=1024,
    )
    return {
        "status": "verified" if backward["success"] else "failed",
        "decomposition": decomposition, "backward": backward,
    }


def generate_trajectory(
    llm, environment, example, system_prompt, trajectory_id, strategy="free",
    max_calls=5, temperature=0.7, max_response_tokens=2048, max_sequence_tokens=16384,
    reference=None, check_redundancy=True, prefix_steps=None, next_action_kind=None,
    seed=None, execution_budget=None, check_provenance=False, proposal_retries=0,
    excluded_action=None, allow_answer_equivalence=False, stop_when_ready=True,
):
    if strategy not in STRATEGIES or max_calls < 1:
        raise ValueError("Invalid strategy or tool budget")
    record = {
        "id": trajectory_id, "trajectory_id": trajectory_id, "question_id": str(example["id"]),
        "question": example["question"], "gold_answers": example.get("golden_answers", []),
        "strategy": strategy, "reference_status": (reference or {}).get("status", "not_requested"),
        "steps": [], "model_responses": [], "success": False, "abort_reason": "",
        "max_tool_calls": max_calls, "quality_check": None, "redundancy_checks": [],
        "stop_protocol": "prefix_v2" if stop_when_ready else "legacy", "readiness_checks": [],
        "conclusion_checks": [],
        "logic_checks": [],
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {example['question']}\nTool-call budget: {max_calls}."},
    ]
    session = environment.session(trajectory_id)
    try:
        if prefix_steps:
            record["steps"] = environment.clone_prefix(prefix_steps, session)
            for step in record["steps"]:
                messages.extend([
                    {"role": "assistant", "content": f"<think>\n{step['think']}\n</think>\n<tool_call>\n{json.dumps(step['action'], ensure_ascii=False)}\n</tool_call>"},
                    {"role": "tool", "content": json.dumps(step["observation"], ensure_ascii=False, separators=(",", ":"))},
                ])
        prefix_length = len(record["steps"])
        for turn in range(prefix_length, max_calls + 1):
            teacher_messages = copy.deepcopy(messages)
            for message, step in zip(teacher_messages[2::2], record["steps"]):
                message["content"] = json.dumps({
                    "reasoning": step["think"], "action": step["action"], "answer": None,
                }, ensure_ascii=False)
            teacher_messages[0]["content"] = (
                "Answer a Wikipedia QA question using real retrieval tools. Corpus file: corpus.jsonl.\n"
                "shell takes command and optional scope (corpus or candidates with candidate_set_id).\n"
                "dense_search takes query only and returns top-3. Shell commands are read-only pipelines;\n"
                "use rg -F or rg -i -F, pipe to head -n 3 to bound output. Counts over candidates are local.\n"
                "Allowed programs: rg, grep, cat, head, tail, wc, sort, cut, uniq, tr, sed, awk.\n"
                + POLICY_INSTRUCTION.replace("tool_call or answer block", "JSON object")
                + "\nReturn ONLY a JSON object, no XML, Markdown or DSML: "
                '{"reasoning":"<your grounded reasoning>",'
                '"action":{"name":"shell or dense_search","arguments":{...}},"answer":null}'
                ' OR {"reasoning":"<your grounded conclusion>","action":null,"answer":"<answer>"}.'
                "\nTool schemas:\n" + json.dumps(tool_schemas(), ensure_ascii=False)
            )
            if strategy != "free":
                restriction = {
                    "dense_first": "For the first retrieval, explore a dense_search query.",
                    "shell_first": "For the first retrieval, explore an exact shell query.",
                    "shell_only": "For this comparison rollout, use only shell with corpus scope.",
                    "dense_only": "For this comparison rollout, use only dense_search.",
                }[strategy]
                if turn == 0 or strategy.endswith("_only"):
                    teacher_messages[0]["content"] += "\nSampling instruction: only if retrieval is necessary; supported answers always take priority. " + restriction
            if next_action_kind and turn == prefix_length:
                restriction = {
                    "dense_search": "For this alternative continuation, make the next action dense_search, grounded in the visible history.",
                    "shell": "For this alternative continuation, make the next action shell with corpus scope, grounded in the visible history.",
                    "shell_candidates": "For this alternative continuation, make the next action shell with candidates scope using an observed handle; recover truncated content or resolve a specific visible ambiguity.",
                    "stop": "For this alternative continuation, answer now using only the visible evidence. State insufficient evidence if the answer is not fully supported.",
                }[next_action_kind]
                teacher_messages[0]["content"] += "\nSampling instruction: only if retrieval is necessary; supported answers always take priority. " + restriction
            if turn == max_calls:
                teacher_messages[-1]["content"] += "\nRetrieval budget is exhausted. Give a supported answer now."
            frontier = example["question"] + "\n" + "\n".join(
                visible_text(step["observation"]) for step in record["steps"] if step["observation"].get("status") == "ok"
            )
            required_kind = None
            if turn == 0 and strategy != "free":
                required_kind = "dense_search" if strategy in ("dense_first", "dense_only") else "shell"
            if turn == max_calls:
                required_kind = "stop"
            if next_action_kind and turn == prefix_length:
                required_kind = next_action_kind
            reasoning, action, answer, failure = propose_step(
                llm, teacher_messages, record, frontier, required_kind, turn, max_calls,
                temperature, max_response_tokens, seed, proposal_retries, check_provenance,
                excluded_action if turn == prefix_length else None,
                "dense_search" if strategy == "dense_only" else "shell" if strategy == "shell_only" else None,
            )
            if failure:
                record["abort_reason"] = failure
                break
            if action is not None and stop_when_ready:
                payload = prefix_payload(example["question"], record["steps"], reasoning, visible_text)
                logical = check_short_circuit(llm, payload, judge_call)
                record["logic_checks"].append({"before_tool": turn + 1, **logical})
                if logical["status"] == "uncertain":
                    record["abort_reason"] = "boolean_logic_uncertain"
                    break
                if logical["status"] == "stop":
                    reasoning, action, answer = logical["reason"], None, logical["answer"]
                else:
                    conclusion = check_conclusion(llm, conclusion_payload(
                        example["question"], record["steps"], reasoning), judge_call)
                    record["conclusion_checks"].append({"before_tool": turn + 1, **conclusion})
                    if conclusion["state"] == "uncertain":
                        record["abort_reason"] = "conclusion_uncertain"
                        break
                    ready = check_ready(llm, payload, judge_call)
                    record["readiness_checks"].append({"before_tool": turn + 1, **ready})
                    if ready.get("status") == "uncertain":
                        record["abort_reason"] = "readiness_uncertain"
                        break
                    if ready.get("status") == "ready":
                        reasoning, action, answer = ready["reason"], None, ready["answer"]
                    elif conclusion["state"] == "complete":
                        record["abort_reason"] = "answer_concluded_without_support"
                        break
            if answer is not None:
                record["final"] = {"think": reasoning, "answer": answer}
                messages.append({"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"})
                break
            if turn == max_calls:
                record["abort_reason"] = "tool_budget_exhausted"
                break
            if next_action_kind and turn == prefix_length:
                actual_kind = "shell_candidates" if action["name"] == "shell" and action["arguments"].get("scope", {}).get("type") == "candidates" else action["name"]
                if actual_kind != next_action_kind:
                    record["abort_reason"] = "branch_action_violation"
                    break
            expected_tool = "dense_search" if strategy in ("dense_only", "dense_first") else "shell"
            if (strategy.endswith("_only") or turn == 0 and strategy != "free") and action["name"] != expected_tool:
                record["abort_reason"] = "sampling_strategy_violation"
                break
            if strategy == "shell_only" and action["arguments"].get("scope", {}).get("type") == "candidates":
                record["abort_reason"] = "sampling_strategy_violation"
                break
            query = action["arguments"].get("command", action["arguments"].get("query", ""))
            if unseen_gold(query, record["gold_answers"], frontier):
                record["abort_reason"] = "unseen_gold_in_query"
                break
            if execution_budget is not None and not execution_budget.consume():
                record["abort_reason"] = "exploration_budget_exhausted"
                break
            observation, metrics = environment.execute(action, session, max_calls - turn - 1)
            record["steps"].append({
                "think": reasoning, "action": action, "observation": observation, "metrics": metrics,
            })
            messages.extend([
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n<tool_call>\n{json.dumps(action, ensure_ascii=False)}\n</tool_call>"},
                {"role": "tool", "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))},
            ])
    finally:
        session.close()
    record["messages"] = messages
    record["mixed"] = len({step["action"]["name"] for step in record["steps"]}) > 1
    record["path"] = "->".join(
        "shell_candidates" if step["action"]["name"] == "shell"
        and step["action"]["arguments"].get("scope", {}).get("type") == "candidates"
        else step["action"]["name"]
        for step in record["steps"]
    )
    if record["abort_reason"] or not record.get("final"):
        record["abort_reason"] = record["abort_reason"] or "missing_final"
        return record
    answer = record["final"]["answer"]
    record["em"] = qa_exact_match(answer, record["gold_answers"])
    record["f1"] = qa_f1_score(answer, record["gold_answers"])
    answer_matches = record["em"]
    if not answer_matches and allow_answer_equivalence:
        record["answer_equivalence"], answer_matches = answer_equivalence(llm, record)
    record["answer_matches"] = answer_matches
    if not answer_matches:
        record["abort_reason"] = "incorrect_answer"
        return record
    if stop_when_ready:
        record["answer_grounding"] = answer_support(record["question"], [visible_text(s["observation"])
            for s in record["steps"] if s["observation"].get("status") == "ok"], answer)
        if not record["answer_grounding"]["supported"]:
            record["abort_reason"] = "unseen_named_final_answer"
            return record
    if stop_when_ready and any(step["observation"].get("status") not in ("ok", "no_hits")
                               for step in record["steps"]):
        record["abort_reason"] = "invalid_tool_trajectory"
        return record
    record["sequence_tokens"] = sequence_length(environment.tokenizer, messages)
    if record["sequence_tokens"] > max_sequence_tokens:
        record["abort_reason"] = "sequence_too_long"
        return record
    if check_provenance:
        review, failure = audit_provenance(llm, record)
        record["provenance_check"] = review
        if failure:
            record["abort_reason"] = failure
            return record
    review_input = {key: record[key] for key in ("question", "gold_answers", "steps", "final")}
    review = judge_call(llm, REVIEW_PROMPT + json.dumps(review_input, ensure_ascii=False), max_tokens=4096)
    record["quality_check"] = review
    failure = validate_review(review, record)
    if failure == "quote_not_in_visible_evidence":
        record["initial_quality_check"] = review
        review = judge_call(llm, REVIEW_PROMPT + json.dumps(review_input, ensure_ascii=False)
                            + "\nYour previous review contained a quote absent from visible contents. "
                            "Review again; copy contiguous exact substrings with original punctuation, "
                            "without inserted ellipses, paraphrases or normalized quotation marks. "
                            "Fail if the required relation has no visible supporting quote.\n"
                            + json.dumps(review, ensure_ascii=False), max_tokens=4096)
        record["quality_check"] = review
        failure = validate_review(review, record)
    if failure:
        record["abort_reason"] = failure
        return record
    if stop_when_ready:
        record["stop_audit"], failure = audit_prefixes(llm, record, judge_call, visible_text)
        if failure:
            record["abort_reason"] = failure
            return record
        record["stop_audit_passed"] = True
        record["conclusion_audit_passed"] = True
        record["logic_audit_passed"] = True
    if check_redundancy and not stop_when_ready:
        contributions = {entry["turn"]: entry["contribution"] for entry in review["turns"]}
        for turn, step in enumerate(record["steps"], 1):
            if step["observation"]["status"] != "ok" or contributions[turn] == "recovery":
                continue
            remaining_evidence = [
                {"turn": index, "contents": visible_text(other["observation"])}
                for index, other in enumerate(record["steps"], 1)
                if index != turn and other["observation"]["status"] == "ok"
            ]
            result = judge_call(llm, ABLATION_PROMPT + json.dumps({
                "question": record["question"], "answer": answer, "remaining_evidence": remaining_evidence,
            }, ensure_ascii=False))
            record["redundancy_checks"].append({"removed_turn": turn, **result})
            if type(result.get("supported")) is not bool:
                record["abort_reason"] = "ablation_review_failed"
                return record
            if result["supported"]:
                record["abort_reason"] = "redundant_observation"
                return record
    record["success"] = True
    return record


def sft_record(record):
    if not record["success"] or leakage_failure(record):
        return None
    if record.get("stop_protocol") in ("prefix_v1", "prefix_v2") and record.get("stop_audit_passed") is not True:
        return None
    if record.get("stop_protocol") == "prefix_v2" and record.get("conclusion_audit_passed") is not True:
        return None
    if record.get("stop_protocol") == "prefix_v2" and record.get("logic_audit_passed") is not True:
        return None
    if record.get("stop_protocol") in ("prefix_v1", "prefix_v2") and not answer_support(record["question"],
            [visible_text(s["observation"]) for s in record["steps"] if s["observation"].get("status") == "ok"],
            record["final"]["answer"])["supported"]:
        return None
    exported = {
        key: record[key] for key in (
            "id", "trajectory_id", "question_id", "question", "strategy", "mixed",
            "path", "em", "f1", "messages",
        )
    } | {"tools": tool_schemas(), "quality_check": record["quality_check"]}
    for key in ("source_dataset", "state_id", "parent_trajectory_id", "compression_parent", "provenance_check", "answer_equivalence", "answer_matches", "stop_protocol", "readiness_checks", "stop_audit", "stop_audit_passed", "answer_grounding", "conclusion_checks", "conclusion_audit_passed", "logic_checks", "logic_audit_passed"):
        if key in record:
            exported[key] = record[key]
    return exported
