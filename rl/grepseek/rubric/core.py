"""Five shared process rubrics: strict grading, statistics and versioned upkeep.

This module has no torch/Ray dependencies. The trainer adapter applies its scalar
bonuses before dynamic group selection; the GSPO loss itself is unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = {"pass", "fail", "not_applicable", "uncertain"}
ITEM_FIELDS = {"id", "revision", "criterion", "applicability", "pass_conditions", "fail_conditions", "exceptions"}


class RubricError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_item(item):
    if not isinstance(item, dict) or set(item) != ITEM_FIELDS:
        raise RubricError("A rubric must have exactly the documented item fields (weights are fixed at 0.1)")
    for key in ("id", "criterion", "applicability"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise RubricError(f"Invalid rubric {key}")
    if type(item["revision"]) is not int or item["revision"] < 0:
        raise RubricError("Invalid rubric revision")
    for key in ("pass_conditions", "fail_conditions", "exceptions"):
        if not isinstance(item[key], list) or any(not isinstance(s, str) or not s.strip() for s in item[key]):
            raise RubricError(f"Invalid rubric {key}")
    if not item["pass_conditions"] or not item["fail_conditions"]:
        raise RubricError("A rubric needs observable pass and fail conditions")
    return copy.deepcopy(item)


def validate_snapshot(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "version", "effective_from_step", "origin", "rubrics"}:
        raise RubricError("Invalid rubric snapshot schema")
    if value["schema_version"] != 1 or type(value["version"]) is not int or value["version"] < 0:
        raise RubricError("Invalid snapshot version")
    if type(value["effective_from_step"]) is not int or value["effective_from_step"] < 1:
        raise RubricError("Invalid effective step")
    if not isinstance(value["origin"], str) or not value["origin"].strip():
        raise RubricError("Snapshot origin is required")
    items = value["rubrics"]
    if not isinstance(items, list) or len(items) != 5:
        raise RubricError("Exactly five active rubrics are required")
    items = [validate_item(item) for item in items]
    if len({item["id"] for item in items}) != 5:
        raise RubricError("Rubric ids must be unique")
    return {**copy.deepcopy(value), "rubrics": items}


def load_snapshot(path):
    return validate_snapshot(json.loads(Path(path).read_text(encoding="utf-8")))


def public_trajectory(record):
    """Whitelist judge-visible fields; never send reference answers or F1."""
    return {"trajectory_id": record["trajectory_id"], "question": record["question"],
            "events": [{key: event[key] for key in ("id", "kind", "text")} for event in record["events"]]}


def visible_events(tokenizer, response_ids, response_mask):
    """Recover the exact delivered token spans, after every observation cutoff.

    Mask-zero spans can also contain environment prefills; they remain explicitly
    observed context. An executed tool whose output was never appended contributes
    no observation. No backend result or reconstructed chat message is consulted.
    """
    if len(response_ids) != len(response_mask) or any(x not in (0, 1) for x in response_mask):
        raise RubricError("Unaligned visible response token mask")
    events = []
    start = 0
    while start < len(response_ids):
        end = start + 1
        while end < len(response_ids) and response_mask[end] == response_mask[start]:
            end += 1
        kind = "action" if response_mask[start] else "observation"
        identifier = f"{'action' if response_mask[start] else 'obs'}_{len(events) + 1}"
        events.append({"id": identifier, "kind": kind,
                       "text": tokenizer.decode(response_ids[start:end], skip_special_tokens=False),
                       "token_start": start, "token_end": end})
        start = end
    if events and events[-1]["kind"] == "action":
        events[-1]["kind"] = "answer"
        events[-1]["id"] = "answer"
    return events


def validate_events(record):
    if not isinstance(record.get("trajectory_id"), str) or not record["trajectory_id"]:
        raise RubricError("Missing trajectory id")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise RubricError("Missing original question")
    events = record.get("events")
    if not isinstance(events, list) or not events:
        raise RubricError("Missing final actor-visible events")
    ids = []
    for event in events:
        if not isinstance(event, dict) or not all(isinstance(event.get(k), str) for k in ("id", "kind", "text")):
            raise RubricError("Invalid visible event")
        if not event["id"] or event["kind"] not in {"question", "action", "observation", "answer"}:
            raise RubricError("Invalid visible event identity")
        ids.append(event["id"])
    if len(ids) != len(set(ids)) or not any(e["kind"] == "answer" and e["text"].strip() for e in events):
        raise RubricError("Duplicate event ids or missing final answer")


def validate_grades(value, snapshot, record):
    if not isinstance(value, dict) or set(value) != {"grades"} or not isinstance(value["grades"], list):
        raise RubricError("Judge must return an object with a grades list")
    events = {event["id"]: event["text"] for event in record["events"]}
    required = {"rubric_id", "status", "evidence", "reason"}
    grades = {}
    for grade in value["grades"]:
        if not isinstance(grade, dict) or set(grade) != required:
            raise RubricError("Invalid rubric grade schema")
        identifier = grade["rubric_id"]
        if identifier in grades or grade["status"] not in STATUSES:
            raise RubricError("Duplicate rubric grade or invalid status")
        if not isinstance(grade["reason"], str) or not grade["reason"].strip() or not isinstance(grade["evidence"], list):
            raise RubricError("Missing grade reason/evidence")
        for evidence in grade["evidence"]:
            if not isinstance(evidence, dict) or set(evidence) != {"event_id", "quote"}:
                raise RubricError("Invalid evidence reference")
            event_id, quote = evidence["event_id"], evidence["quote"]
            if event_id not in events or not isinstance(quote, str) or not quote.strip() or quote not in events[event_id]:
                raise RubricError("Judge evidence is not a verbatim quote of a visible event")
        if grade["status"] in {"pass", "fail"} and not grade["evidence"]:
            raise RubricError("Clear grades require visible evidence")
        grades[identifier] = copy.deepcopy(grade)
    if set(grades) != {item["id"] for item in snapshot["rubrics"]}:
        raise RubricError("Judge must grade exactly the five active rubric ids")
    return {item["id"]: grades[item["id"]] for item in snapshot["rubrics"]}


def combine_group_rewards(records, grades):
    """Return scalar rewards; technical failure falls back for the entire group.

    For a semantic uncertain/N/A grade, mask that item across all valid members
    of the group. This does not turn an unavailable judgment into a failure.
    Invalid trajectories always keep their baseline format penalty.
    """
    if len(records) != len(grades):
        raise RubricError("Unaligned trajectory and grade lists")
    valid = [i for i, record in enumerate(records) if record["valid"]]
    technical_fallback = any(grades[i] is None for i in valid)
    ids = list(grades[valid[0]]) if valid and not technical_fallback else []
    masks = {k: not technical_fallback and all(grades[i][k]["status"] in {"pass", "fail"} for i in valid) for k in ids}
    results = []
    for i, record in enumerate(records):
        bonus = 0.0
        passed = 0
        if record["valid"] and not technical_fallback:
            passed = sum(masks[k] and grades[i][k]["status"] == "pass" for k in ids)
            bonus = .1 * passed
        results.append({"score": record["base_score"] + bonus, "bonus": bonus, "passed": passed,
                        "fallback": bool(technical_fallback and record["valid"]),
                        "masked_items": sum(not flag for flag in masks.values())})
    return results


def summarize(records, snapshot):
    """All-sampled statistics, before reward selection; questions are balanced."""
    result = {"trajectories": len(records), "question_groups": len({r["group_id"] for r in records}),
              "distinct_questions": len({r["qid"] for r in records}),
              "format_invalid": sum(r.get("format_invalid", not r["valid"]) for r in records),
              "failed_rollouts": sum(r.get("failed_rollout", False) for r in records),
              "judge_errors": sum(r["valid"] and r.get("grades") is None for r in records), "items": {}}
    for item in snapshot["rubrics"]:
        counts = Counter({s: 0 for s in STATUSES})
        by_question, by_group, buckets = defaultdict(list), defaultdict(list), defaultdict(Counter)
        eligible = 0
        applicable_questions = set()
        for record in records:
            if not record["valid"] or record.get("grades") is None:
                continue
            eligible += 1
            status = record["grades"][item["id"]]["status"]
            counts[status] += 1
            if status != "not_applicable":
                applicable_questions.add(record["qid"])
            for bucket in record.get("buckets", []):
                buckets[bucket][status] += 1
            if status in {"pass", "fail"}:
                by_question[record["qid"]].append(status == "pass")
                by_group[record["group_id"]].append(status == "pass")
        groups = [labels for labels in by_group.values() if len(labels) >= 2]
        clear = counts["pass"] + counts["fail"]
        applicable = clear + counts["uncertain"]
        result["items"][item["id"]] = {
            "revision": item["revision"], **dict(counts), "clear": clear,
            "distinct_questions": len(by_question),
            "applicable_questions": len(applicable_questions),
            "raw_pass_rate": counts["pass"] / clear if clear else None,
            "pass_rate": sum(sum(v) / len(v) for v in by_question.values()) / len(by_question) if by_question else None,
            "group_discrimination": sum(any(v) and not all(v) for v in groups) / len(groups) if groups else None,
            "discrimination_groups": len(groups),
            "uncertainty_rate": counts["uncertain"] / applicable if applicable else None,
            "applicability_rate": applicable / eligible if eligible else None,
            "buckets": {k: dict(v) for k, v in buckets.items()},
        }
    return result


def load_calibration(path, snapshot):
    """A separate training-only reference panel, never final validation/test data."""
    if not path:
        return []
    records, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        validate_events(record)
        if record.get("split") != "train" or not isinstance(record.get("qid"), str):
            raise RubricError(f"Calibration row {number} must have split=train and a qid")
        if record["trajectory_id"] in seen:
            raise RubricError("Repeated calibration trajectory")
        seen.add(record["trajectory_id"])
        refs = record.get("reference_statuses", {})
        if set(refs) != {r["id"] for r in snapshot["rubrics"]} or any(v not in STATUSES for v in refs.values()):
            raise RubricError("Calibration needs five explicit reference statuses")
        if record.get("initial_snapshot_hash") != digest(snapshot):
            raise RubricError("Calibration references belong to another initial snapshot")
        if not isinstance(record.get("reference_source"), str) or not record["reference_source"]:
            raise RubricError("Calibration reference provenance is required")
        records.append(record)
    return records


DEFAULT_MAINTENANCE = dict(min_clear=64, min_questions=32, consecutive_checks=3,
                           high_pass_rate=.95, low_pass_rate=.10, low_discrimination=.05,
                           high_uncertainty=.20, cooldown_steps=3, max_examples=40,
                           min_calibration_questions=8, max_calibration_examples=32,
                           min_agreement=.90, max_false_penalty_rate=.05,
                           statistics_window_steps=20)


def compact_evidence(records, snapshot, statistics, step):
    """Small cross-step evidence, without storing observations in checkpoints."""
    result = {"step": step, "items": {}}
    for item in snapshot["rubrics"]:
        per_question = defaultdict(lambda: [0, 0])
        applicable_qids = set()
        for record in records:
            if not record["valid"] or record.get("grades") is None:
                continue
            status = record["grades"][item["id"]]["status"]
            if status != "not_applicable":
                applicable_qids.add(record["qid"])
            if status in {"pass", "fail"}:
                per_question[record["qid"]][int(status == "fail")] += 1
        result["items"][item["id"]] = {
            "revision": item["revision"], "question_counts": dict(per_question),
            "applicable_qids": sorted(applicable_qids),
            "statistics": copy.deepcopy(statistics["items"][item["id"]]),
        }
    return result


def window_statistics(history, snapshot):
    """Pool only matching item revisions; repeated qids retain one question weight."""
    result = {"window_steps": len(history), "items": {}}
    for item in snapshot["rubrics"]:
        entries = [h["items"][item["id"]] for h in history
                   if h["items"][item["id"]]["revision"] == item["revision"]]
        questions = defaultdict(lambda: [0, 0])
        applicable = set()
        counts = Counter({s: 0 for s in STATUSES})
        discrimination_groups = 0
        mixed_groups = 0.0
        for entry in entries:
            for qid, (passed, failed) in entry["question_counts"].items():
                questions[qid][0] += passed
                questions[qid][1] += failed
            applicable.update(entry["applicable_qids"])
            stats = entry["statistics"]
            counts.update({s: stats[s] for s in STATUSES})
            discrimination_groups += stats["discrimination_groups"]
            mixed_groups += (stats["group_discrimination"] or 0) * stats["discrimination_groups"]
        clear = counts["pass"] + counts["fail"]
        judged_applicable = clear + counts["uncertain"]
        result["items"][item["id"]] = {
            "revision": item["revision"], **dict(counts), "clear": clear,
            "distinct_questions": len(questions), "applicable_questions": len(applicable),
            "pass_rate": sum(p / (p + f) for p, f in questions.values()) / len(questions) if questions else None,
            "group_discrimination": mixed_groups / discrimination_groups if discrimination_groups else None,
            "discrimination_groups": discrimination_groups,
            "uncertainty_rate": counts["uncertain"] / judged_applicable if judged_applicable else None,
            "evidence_steps": len(entries),
        }
    return result


class RubricWorkflow:
    """Synchronous, checkpointable maintenance. A failed proposal preserves G_t."""

    def __init__(self, initial, output_dir, client, *, calibration_path="", maintenance=None):
        self.initial = validate_snapshot(initial)
        self.active = copy.deepcopy(self.initial)
        self.output_dir = Path(output_dir)
        self.client = client
        self.options = {**DEFAULT_MAINTENANCE, **(maintenance or {})}
        if set(self.options) != set(DEFAULT_MAINTENANCE):
            raise RubricError("Unknown rubric maintenance option")
        for key in ("min_clear", "min_questions", "consecutive_checks", "max_examples", "min_calibration_questions", "max_calibration_examples", "statistics_window_steps"):
            if type(self.options[key]) is not int or self.options[key] < 1:
                raise RubricError(f"Invalid maintenance option: {key}")
        if type(self.options["cooldown_steps"]) is not int or self.options["cooldown_steps"] < 0:
            raise RubricError("Invalid cooldown")
        if self.options["max_calibration_examples"] < self.options["min_calibration_questions"]:
            raise RubricError("Reference budget is smaller than the required question coverage")
        for key in ("high_pass_rate", "low_pass_rate", "low_discrimination", "high_uncertainty", "min_agreement", "max_false_penalty_rate"):
            if not 0 <= self.options[key] <= 1:
                raise RubricError(f"Invalid maintenance threshold: {key}")
        self.calibration = load_calibration(calibration_path, self.initial)
        self.calibration_hash = digest(self.calibration)
        self.completed_step = 0
        self.counters = {}
        self.last_changes = {}
        self.history = []
        self.frozen = None
        self.step = None
        self.records = []
        self._seen_groups = set()
        self._save_snapshot(self.initial)

    def _save_snapshot(self, snapshot):
        path = self.output_dir / "versions" / f"v{snapshot['version']:06d}-{digest(snapshot)}.json"
        if path.exists() and json.loads(path.read_text()) != snapshot:
            raise RubricError("Immutable snapshot collision")
        if not path.exists():
            atomic_json(path, snapshot)

    def begin_step(self, step):
        if self.step is not None or step != self.completed_step + 1 or self.active["effective_from_step"] > step:
            raise RubricError("Rubric outer steps must be sequential and non-overlapping")
        self.step = step
        self.frozen = copy.deepcopy(self.active)
        self.records = []
        self._seen_groups = set()
        self.attempt_id = uuid.uuid4().hex
        self.step_dir = self.output_dir / "steps" / f"{step:06d}-{self.attempt_id}"
        atomic_json(self.step_dir / "used_snapshot.json", self.frozen)
        return copy.deepcopy(self.frozen)

    async def score_groups(self, records):
        if self.step is None:
            raise RubricError("begin_step must precede scoring")
        group_ids = {r["group_id"] for r in records}
        if self._seen_groups & group_ids:
            raise RubricError("A question group cannot be scored twice in an outer step")
        valid_records = []
        for record in records:
            if not math.isfinite(record["base_score"]):
                raise RubricError("Nonfinite base reward")
            if record["valid"]:
                validate_events(record)
                if not 0 <= record["f1"] <= 1 or not math.isclose(record["base_score"], record["f1"], abs_tol=1e-6):
                    raise RubricError("Rubric reward requires raw F1; disable other scalar reward additions")
                valid_records.append(record)
        values = await self.client.grade_many(self.frozen, valid_records)
        grade_by_id = dict(zip((r["trajectory_id"] for r in valid_records), values, strict=True))
        groups = defaultdict(list)
        for record in records:
            groups[record["group_id"]].append(record)
        outputs = {}
        for group in groups.values():
            grades = [grade_by_id.get(r["trajectory_id"]) for r in group]
            for record, grade, score in zip(group, grades, combine_group_rewards(group, grades), strict=True):
                saved = {**copy.deepcopy(record), "grades": grade, **score,
                         "step": self.step, "split": "train",
                         "rubric_version": self.frozen["version"], "snapshot_hash": digest(self.frozen)}
                self.records.append(saved)
                outputs[record["trajectory_id"]] = saved
        self._seen_groups.update(group_ids)
        with (self.step_dir / "all_sampled.jsonl").open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(outputs[record["trajectory_id"]], ensure_ascii=False, allow_nan=False) + "\n")
        return [outputs[r["trajectory_id"]] for r in records]

    def _eligible(self, statistics):
        eligible = []
        for item in self.frozen["rubrics"]:
            key = f"{item['id']}:{item['revision']}"
            stats = statistics["items"][item["id"]]
            p, d = stats["pass_rate"], stats["group_discrimination"]
            reason = None
            uncertainty_mature = (stats["clear"] + stats["uncertain"] >= self.options["min_clear"]
                                  and stats["applicable_questions"] >= self.options["min_questions"])
            clear_mature = stats["clear"] >= self.options["min_clear"] and stats["distinct_questions"] >= self.options["min_questions"]
            if uncertainty_mature and stats["uncertainty_rate"] is not None and stats["uncertainty_rate"] > self.options["high_uncertainty"]:
                reason = "ambiguous"
            elif not clear_mature:
                continue
            elif p >= self.options["high_pass_rate"] and d is not None and d <= self.options["low_discrimination"]:
                reason = "possibly_saturated"
            elif p <= self.options["low_pass_rate"]:
                reason = "diagnose_failure_not_remove_target"
            elif d is not None and d <= self.options["low_discrimination"]:
                reason = "inspect_between_question_only_signal"
            old = self.counters.get(key, {})
            count = old.get("count", 0) + 1 if reason and old.get("reason") == reason else int(bool(reason))
            self.counters[key] = {"reason": reason, "count": count}
            cooldown = self.step - self.last_changes.get(item["id"], -1000000)
            if reason and count >= self.options["consecutive_checks"] and cooldown >= self.options["cooldown_steps"]:
                eligible.append({"rubric_id": item["id"], "reason": reason, "statistics": stats})
        return sorted(eligible, key=lambda e: self.last_changes.get(e["rubric_id"], -1000000))

    async def finish_step(self, retained_ids):
        if self.step is None:
            raise RubricError("No active rubric step")
        retained_ids = set(retained_ids)
        all_ids = {r["trajectory_id"] for r in self.records}
        if not retained_ids <= all_ids:
            raise RubricError("Unknown retained trajectory")
        statistics = summarize(self.records, self.frozen)
        statistics["retained_trajectories"] = len(retained_ids)
        statistics["retained_question_groups"] = len({r["group_id"] for r in self.records if r["trajectory_id"] in retained_ids})
        self.history.append(compact_evidence(self.records, self.frozen, statistics, self.step))
        self.history = self.history[-self.options["statistics_window_steps"]:]
        maintenance_statistics = window_statistics(self.history, self.frozen)
        eligible = self._eligible(maintenance_statistics)
        decision = {"action": "KEEP", "reason": "no_mature_trigger", "eligible": eligible}
        if eligible:
            calibration_qids = {r["qid"] for r in self.calibration}
            examples = [r for r in self.records if r["valid"] and r.get("grades") is not None and r["qid"] not in calibration_qids]
            # Preserve within-question contrasts for the maintainer. At most
            # two distinct grade patterns per qid keep large groups from taking
            # the whole proposal budget; full events remain in all_sampled.jsonl.
            examples.sort(key=lambda r: digest((r["qid"], r["trajectory_id"])))
            by_qid = defaultdict(dict)
            for record in examples:
                pattern = tuple(g["status"] for g in record["grades"].values())
                by_qid[record["qid"]].setdefault(pattern, record)
            selected = []
            eligible_ids = {e["rubric_id"] for e in eligible}
            def priority(patterns):
                relevant = any(r["grades"][k]["status"] != "not_applicable"
                               for r in patterns.values() for k in eligible_ids)
                return (not relevant, -len(patterns))
            for patterns in sorted(by_qid.values(), key=priority):
                pair = list(patterns.values())[:2]
                if len(selected) + len(pair) <= self.options["max_examples"]:
                    selected.extend(pair)
            if len(calibration_qids) < self.options["min_calibration_questions"]:
                decision["reason"] = "missing_training_reference_panel"
            elif not selected or all(r["grades"][k]["status"] == "not_applicable" for r in selected for k in eligible_ids):
                decision["reason"] = "no_proposal_cases_disjoint_from_calibration"
            else:
                try:
                    proposal = await self.client.propose(self.frozen, eligible, selected)
                    decision = await self._validate_proposal(proposal, eligible, selected)
                except RubricError as exc:
                    decision = {"action": "KEEP", "reason": "proposal_or_validation_failed", "error": str(exc)}
        self.completed_step = self.step
        atomic_json(self.step_dir / "summary.json", {"step": self.step, "used_version": self.frozen["version"],
                    "next_version": self.active["version"], "statistics": statistics,
                    "maintenance_statistics": maintenance_statistics, "maintenance": decision})
        atomic_json(self.output_dir / "state.json", self.state_dict())
        self.step = None
        self.frozen = None
        self.records = []
        return statistics, decision

    async def _validate_proposal(self, proposal, eligible, examples):
        if not isinstance(proposal, dict) or proposal.get("action") not in {"KEEP", "REVISE", "REPLACE"}:
            raise RubricError("Invalid maintenance action")
        if proposal["action"] == "KEEP":
            return {"action": "KEEP", "reason": str(proposal.get("reason", "maintainer_keep"))}
        if set(proposal) != {"action", "rubric_id", "item", "reason", "case_ids"}:
            raise RubricError("A proposal must change exactly one rubric with cited cases")
        identifier = proposal["rubric_id"]
        if not isinstance(identifier, str) or identifier not in {e["rubric_id"] for e in eligible}:
            raise RubricError("Proposal targets a rubric without a mature trigger")
        item = validate_item(proposal["item"])
        old = next(r for r in self.frozen["rubrics"] if r["id"] == identifier)
        if item["id"] != identifier or item["revision"] != old["revision"] + 1:
            raise RubricError("A changed item must retain its id and increment its revision")
        if {k:v for k,v in item.items() if k != 'revision'} == {k:v for k,v in old.items() if k != 'revision'}:
            raise RubricError("A revision must change an observable rubric definition")
        available = {r["trajectory_id"] for r in examples}
        if (not isinstance(proposal["case_ids"], list) or not proposal["case_ids"]
                or any(not isinstance(k, str) for k in proposal["case_ids"])
                or not set(proposal["case_ids"]) <= available):
            raise RubricError("Proposal cites unavailable examples")
        if not isinstance(proposal["reason"], str) or not proposal["reason"].strip():
            raise RubricError("Proposal needs a reason")
        candidate = copy.deepcopy(self.frozen)
        candidate["version"] += 1
        candidate["effective_from_step"] = self.step + 1
        candidate["rubrics"] = [item if r["id"] == identifier else r for r in candidate["rubrics"]]
        candidate = validate_snapshot(candidate)
        initial_item = next(r for r in self.initial["rubrics"] if r["id"] == identifier)
        review = await self.client.review(initial_item, old, item)
        required = {"preserves_core", "tool_neutral", "answer_free", "reason"}
        if (not isinstance(review, dict) or set(review) != required
                or any(type(review[k]) is not bool for k in required - {"reason"})
                or not isinstance(review["reason"], str) or not review["reason"].strip()):
            raise RubricError("Invalid semantic review")
        if not all(review[k] for k in required - {"reason"}):
            return {"action": "KEEP", "reason": "semantic_review_rejected", "review": review, "proposal": proposal}
        panel = []
        per_qid = set()
        queues = {status:[r for r in self.calibration if r["reference_statuses"][identifier] == status]
                  for status in ("pass", "fail")}
        while any(queues.values()) and len(panel) < self.options["max_calibration_examples"]:
            for queue in queues.values():
                while queue and queue[0]["qid"] in per_qid:
                    queue.pop(0)
                if queue and len(panel) < self.options["max_calibration_examples"]:
                    record = queue.pop(0)
                    panel.append(record)
                    per_qid.add(record["qid"])
        if len(per_qid) < self.options["min_calibration_questions"]:
            return {"action": "KEEP", "reason": "insufficient_clear_reference_labels", "proposal": proposal}
        if {r["reference_statuses"][identifier] for r in panel} != {"pass", "fail"}:
            return {"action": "KEEP", "reason": "reference_panel_needs_both_pass_and_fail", "proposal": proposal}
        grades = await self.client.grade_many(candidate, panel)
        agreed = false_penalties = positive_refs = 0
        for record, grade in zip(panel, grades, strict=True):
            if grade is None or grade[identifier]["status"] not in {"pass", "fail"}:
                return {"action": "KEEP", "reason": "uncertain_candidate_calibration", "proposal": proposal}
            expected = record["reference_statuses"][identifier]
            actual = grade[identifier]["status"]
            agreed += actual == expected
            positive_refs += expected == "pass"
            false_penalties += expected == "pass" and actual == "fail"
        agreement = agreed / len(panel)
        false_penalty = false_penalties / positive_refs if positive_refs else 0.0
        calibration = {"agreement": agreement, "false_penalty_rate": false_penalty,
                       "questions": len(per_qid), "reference_sources": sorted({r["reference_source"] for r in panel})}
        if agreement < self.options["min_agreement"] or false_penalty > self.options["max_false_penalty_rate"]:
            return {"action": "KEEP", "reason": "reference_panel_rejected", "calibration": calibration, "proposal": proposal}
        self.active = candidate
        self.last_changes[identifier] = self.step
        self._save_snapshot(candidate)
        return {"action": proposal["action"], "reason": proposal["reason"], "proposal": proposal,
                "review": review, "calibration": calibration, "effective_from_step": self.step + 1}

    def state_dict(self):
        return {"schema_version": 1, "initial_hash": digest(self.initial), "calibration_hash": self.calibration_hash,
                "judge_signature": copy.deepcopy(getattr(self.client, "signature", {})),
                "maintenance_options": self.options, "completed_step": self.completed_step,
                "active": self.active, "counters": self.counters, "last_changes": self.last_changes,
                "history": self.history}

    def abort_step(self, reason):
        """Archive an untrained partial rollout without advancing rubric state."""
        if self.step is None:
            raise RubricError("No active rubric step to abort")
        atomic_json(self.step_dir / "aborted.json", {
            "sample_step": self.step, "completed_step": self.completed_step,
            "reason": str(reason), "parameters_updated": False,
            "used_version": self.frozen["version"], "statistics": summarize(self.records, self.frozen),
        })
        self.step = None
        self.frozen = None
        self.records = []
        self._seen_groups = set()

    def restore(self, value, completed_step):
        if value.get("schema_version") != 1 or value.get("initial_hash") != digest(self.initial):
            raise RubricError("Rubric checkpoint initial snapshot mismatch")
        if value.get("calibration_hash") != self.calibration_hash or value.get("maintenance_options") != self.options:
            raise RubricError("Rubric checkpoint reference panel or maintenance configuration mismatch")
        if value.get("judge_signature") != getattr(self.client, "signature", {}):
            raise RubricError("Rubric checkpoint judge model or scoring configuration mismatch")
        if value.get("completed_step") != completed_step:
            raise RubricError("Policy and rubric checkpoints have different completed steps")
        active = validate_snapshot(value["active"])
        if active["effective_from_step"] > completed_step + 1:
            raise RubricError("Checkpoint rubric has not yet become effective")
        self.active = active
        self.completed_step = completed_step
        self.counters = copy.deepcopy(value["counters"])
        self.last_changes = copy.deepcopy(value["last_changes"])
        self.history = copy.deepcopy(value.get("history", []))
        self._save_snapshot(active)
