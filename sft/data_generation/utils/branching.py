from __future__ import annotations

import copy
import hashlib
import json
import time

from .hybrid_pipeline import generate_trajectory, judge_call, visible_text
from .hybrid_prompts import ABLATION_PROMPT


class ExecutionBudget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.active_limit = limit

    def consume(self):
        if self.used >= self.active_limit:
            return False
        self.used += 1
        return True


def state_digest(question, steps):
    normalized = copy.deepcopy(steps)
    handles = {}
    for step in normalized:
        step.pop("metrics", None)
        observation = step["observation"]
        if "candidate_set_id" in observation:
            handle = observation["candidate_set_id"]
            handles[handle] = f"candidate_{len(handles)}"
            observation["candidate_set_id"] = handles[handle]
        scope = step["action"]["arguments"].get("scope", {})
        if "candidate_set_id" in scope:
            scope["candidate_set_id"] = handles.get(scope["candidate_set_id"], "unknown")
    return hashlib.sha256(json.dumps({"question": question, "steps": normalized}, sort_keys=True).encode()).hexdigest()


def action_kind(step):
    action = step["action"]
    if action["name"] == "shell" and action["arguments"].get("scope", {}).get("type") == "candidates":
        return "shell_candidates"
    return action["name"]


def select_states(records, max_calls, limit=2):
    candidates = {}
    answered = {state_digest(record["question"], record["steps"])
                for record in records if record.get("success")}
    for record in records:
        for count in range(1, min(len(record["steps"]), max_calls - 1) + 1):
            prefix = record["steps"][:count]
            observation = prefix[-1]["observation"]
            digest = state_digest(record["question"], prefix)
            if digest in answered:
                continue
            score = 3 if observation.get("status") != "ok" else 2 if observation.get("observation_truncated") or observation.get("truncated") else 1
            if record.get("success") and count == len(record["steps"]):
                score = 0
            existing = candidates.get(digest)
            if existing is None or score > existing[0]:
                candidates[digest] = (score, record, count)
    ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], item[1][2], item[0]))
    return [(digest, record, count) for digest, (_, record, count) in ordered[:limit]]


def run_question(llm, environment, example, system_prompt, strategies, max_calls=5,
                 temperature=0.7, max_sequence_tokens=16384, exploration_calls=40,
                 max_states=2, seed=20261012):
    budget = ExecutionBudget(exploration_calls)
    budget.active_limit = max(5 * max_calls, exploration_calls - max_calls)
    budget.active_limit = min(budget.active_limit, exploration_calls)
    records, states, comparisons = [], [], []
    question_id = str(example["id"])
    def run(identifier, strategy="free", prefix=None, kind=None, excluded=None):
        started = time.monotonic()
        record = generate_trajectory(
            llm, environment, example, system_prompt, identifier, strategy,
            max_calls, temperature, max_sequence_tokens=max_sequence_tokens,
            check_redundancy=False, check_provenance=True, prefix_steps=prefix,
            next_action_kind=kind, execution_budget=budget,
            seed=seed + len(records) * 97, proposal_retries=2,
            max_response_tokens=1024, excluded_action=excluded,
            allow_answer_equivalence=True, stop_when_ready=True,
        )
        record["elapsed_seconds"] = time.monotonic() - started
        record["executed_calls"] = len(record["steps"]) - len(prefix or [])
        record["branch_action_kind"] = kind
        record["source_dataset"] = example.get("source_dataset")
        records.append(record)
        return record
    for strategy in strategies:
        run(f"{question_id}::{strategy}::0", strategy)
    for digest, parent, count in select_states(records, max_calls, max_states):
        prefix = parent["steps"][:count]
        kinds = ["shell", "dense_search", "stop"]
        if any(step["observation"].get("candidate_set_id") for step in prefix):
            kinds.insert(2, "shell_candidates")
        state = {"state_id": digest, "question_id": question_id,
                 "parent_trajectory_id": parent["trajectory_id"], "prefix_length": count,
                 "remaining_budget": max_calls - count, "prefix_steps": prefix, "actions": kinds}
        states.append(state)
        outcomes = []
        for kind in kinds:
            if budget.used >= budget.active_limit and kind != "stop":
                outcomes.append({"kind": kind, "status": "not_expanded"})
                continue
            branch = run(f"{question_id}::branch::{digest[:12]}::{kind}", prefix=prefix, kind=kind)
            branch["state_id"] = digest
            branch["parent_trajectory_id"] = parent["trajectory_id"]
            outcomes.append({"kind": kind, "trajectory_id": branch["trajectory_id"],
                             "success": branch["success"], "em": branch.get("em", False),
                             "calls": branch["executed_calls"], "abort_reason": branch["abort_reason"],
                             "status": "incomplete_budget" if branch["abort_reason"] == "exploration_budget_exhausted" else "executed"})
        comparisons.append({"state_id": digest, "question_id": question_id,
                            "remaining_budget": max_calls - count, "outcomes": outcomes})
    budget.active_limit = budget.limit
    compressible = sorted(
        [record for record in records if record.get("answer_matches") and len(record["steps"]) > 1
         and record.get("abort_reason") in ("", "redundant_or_unreviewed_turn")],
        key=lambda item: (not item["mixed"], len(item["steps"]), item["trajectory_id"]),
    )
    compression = []
    for parent in compressible[:1]:
        for index, step in enumerate(parent["steps"]):
            if step["observation"].get("status") != "ok":
                continue
            remaining = [{"turn": turn + 1, "contents": visible_text(other["observation"])}
                         for turn, other in enumerate(parent["steps"]) if turn != index and other["observation"].get("status") == "ok"]
            result = judge_call(llm, ABLATION_PROMPT + json.dumps({"question": example["question"],
                "answer": parent["final"]["answer"], "remaining_evidence": remaining}, ensure_ascii=False))
            item = {"parent_trajectory_id": parent["trajectory_id"], "removed_turn": index + 1, "diagnostic": result}
            if result.get("supported") is True and budget.used < budget.limit:
                replay = run(f"{question_id}::compression::{index}", prefix=parent["steps"][:index], excluded=step["action"])
                replay["compression_parent"] = parent["trajectory_id"]
                item["replay_trajectory_id"] = replay["trajectory_id"]
                item["replay_success"] = replay["success"]
                item["replay_calls"] = len(replay["steps"])
                item["reduced_valid_trajectory"] = replay["success"] and len(replay["steps"]) < len(parent["steps"])
                compression.append(item)
                break
            compression.append(item)
    # A validated STOP at this exact history dominates every longer continuation
    # from that history. Path diversity cannot revive the longer positive.
    supported_stops = {state_digest(record["question"], record["steps"]): record["trajectory_id"]
                       for record in records if record["success"]}
    for record in records:
        if not record["success"]:
            continue
        for count in range(len(record["steps"])):
            digest = state_digest(record["question"], record["steps"][:count])
            if digest in supported_stops:
                record["success"] = False
                record["abort_reason"] = "dominated_by_supported_stop"
                record["supported_stop_trajectory_id"] = supported_stops[digest]
                break
    successful = [record for record in records if record["success"]]
    selected, seen = [], set()
    shortest = sorted(successful, key=lambda item: (len(item["steps"]), item.get("sequence_tokens", 0), item["trajectory_id"]))
    mixed = [record for record in shortest if record["mixed"]]
    ordered = shortest[:1] + mixed[:1] + shortest
    for record in ordered:
        signature = record["path"]
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(record["trajectory_id"])
        if len(selected) == 3:
            break
    for record in records:
        record["selected"] = record["trajectory_id"] in selected
    return {"question_id": question_id, "records": records, "states": states,
            "counterfactuals": comparisons, "executed_calls": budget.used,
            "selected_trajectories": selected, "compression": compression}
