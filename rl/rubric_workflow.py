#!/usr/bin/env python3
"""Offline rubric preparation/replay. None of these commands starts policy RL."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path

import httpx

from grepseek.rubric.core import (
    RubricError, RubricWorkflow, atomic_json, digest, load_calibration, load_snapshot,
    validate_events, validate_snapshot,
)
from grepseek.rubric.client import RubricClient

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "rl/grepseek/trainer/config/rubric/initial_rubrics.json"


def read_jsonl(path):
    return list(iter_jsonl(path))


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def export_inputs(args):
    """Convert archived training outputs; do not reconstruct hidden tool results."""
    if 'val_rollouts' in Path(args.input).parts:
        raise RubricError("Use archived training rollouts, not validation rollouts")
    questions = {str(row.get('qid',row.get('id',''))):row.get('question',row.get('query',''))
                 for row in iter_jsonl(args.train_file)}
    records=[]
    skipped=0
    for index,row in enumerate(iter_jsonl(args.input)):
        gt=row.get('gts') or {}
        if isinstance(gt,str):
            try:gt=json.loads(gt)
            except ValueError:gt={}
        qid=str(row.get('qid') or (gt.get('qid') if isinstance(gt,dict) else '') or '')
        if qid not in questions or row.get('is_validation',False):
            skipped+=1;continue
        text=row.get('output','')
        if not isinstance(text,str) or row.get('reward/format_pass') != 1 or row.get('reward/failed_rollout',0):
            skipped+=1;continue
        answer=re.search(r'<answer>\s*(.*?)\s*</answer>\s*$',text,re.S)
        observations=list(re.finditer(r'<tool_response>.*?</tool_response>',text,re.S))
        if not answer or not answer.group(1).strip() or len(observations)!=text.count('<tool_response>') or len(observations)!=text.count('</tool_response>'):
            skipped+=1;continue
        events=[{'id':'question','kind':'question','text':questions[qid]}]
        start=0
        for number,match in enumerate(observations,1):
            if text[start:match.start()]:
                events.append({'id':f'action_{number}','kind':'action','text':text[start:match.start()]})
            events.append({'id':f'obs_{number}','kind':'observation','text':match.group()})
            start=match.end()
        events.append({'id':'answer','kind':'answer','text':text[start:]})
        f1=float(row['reward/f1'])
        record={'trajectory_id':f'archive-{digest(str(Path(args.input).resolve()))[:12]}-{index}',
                'qid':qid,'group_id':f"archive:{row.get('step',0)}:{qid}",
                'question':questions[qid],'events':events,'valid':True,'f1':f1,'base_score':f1,
                'split':'train','step':1,'source_step':row.get('step'),
                'visibility_source':'archived_decoded_rollout_output;not_backend_reconstruction'}
        validate_events(record)
        records.append(record)
        if len(records)>=args.limit:break
    if not records:
        raise RubricError("No valid archived training trajectories could be exported")
    path=Path(args.output)
    if path.exists():raise RubricError("Refusing to overwrite exported trajectories")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(json.dumps({'exported':len(records),'distinct_training_questions':len({r['qid'] for r in records}),
                      'skipped':skipped,'output':str(path),'policy_training_started':False}))


def client_for(args):
    return RubricClient(dict(base_url=args.base_url, model=args.model, concurrency=args.concurrency,
                             api_key_env="GREPSEEK_RUBRIC_API_KEY", retries=2, timeout=90,
                             audit_path=str(Path(args.output).parent / "preparation_requests.jsonl")))


def check(args):
    snapshot = load_snapshot(args.initial)
    panel = load_calibration(args.calibration, snapshot)
    questions = len({r["qid"] for r in panel})
    coverage = {item['id']: {
        status: len({r['qid'] for r in panel if r['reference_statuses'][item['id']] == status})
        for status in ('pass','fail')} for item in snapshot['rubrics']}
    ready = questions >= 8 and all(c['pass'] and c['fail'] and c['pass']+c['fail'] >= 8 for c in coverage.values())
    if args.require_calibration and not ready:
        raise RubricError("Dynamic publication requires a separate training reference panel with at least 8 qids; prepare-calibration can create an explicitly model-labeled panel")
    if args.live:
        import httpx

        secret = os.environ.get("GREPSEEK_RUBRIC_API_KEY", "")
        if not secret:
            raise RubricError("Missing GREPSEEK_RUBRIC_API_KEY")
        with httpx.Client(timeout=10, trust_env=False) as http:
            response = http.get(args.base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + secret})
            response.raise_for_status()
            if args.model not in {m["id"] for m in response.json()["data"]}:
                raise RubricError("Requested rubric model is not served at the supplied endpoint")
    result = {"active_rubrics": 5, "pass_reward": .1, "valid_reward_range": [0, 1.5],
              "snapshot_hash": digest(snapshot), "initial_origin": snapshot["origin"],
              "reference_questions": questions, "dynamic_publication_ready": ready,
              "reference_label_coverage": coverage,
              "reference_sources": sorted({r["reference_source"] for r in panel}),
              "network_checked": args.live, "policy_training_started": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


async def initialize(args):
    if Path(args.output).exists():
        raise RubricError("Refusing to overwrite an initial rubric snapshot; use a new version path")
    if args.question_limit < 1:
        raise RubricError("question-limit must be positive")
    template = load_snapshot(args.initial)
    rows = read_jsonl(args.train_file)
    selected = random.Random(42).sample(rows, min(len(rows), args.question_limit))
    questions = [{"qid": str(r.get("qid", r.get("id", i))), "question": r.get("question", r.get("query", ""))}
                 for i, r in enumerate(selected)]
    if not questions or any(not q["question"] for q in questions):
        raise RubricError("Initialization requires nonempty training questions")
    client = client_for(args)
    prompt = ("Generate exactly five reusable, answer-free SearchQA process rubrics from training questions. "
              "Preserve the five template ids and revision=0; improve their observable definitions, conditions and exceptions. "
              "Return only {rubrics: [five complete items]}, with exactly the same item fields as the template. "
              "Do not prescribe a fixed tool order or mandatory tool mixing. Distinguish scope/no_hits, errors, truncation, "
              "entity identity, evidence linkage, necessary recovery and justified stopping. Tool interface: frozen dense top-3; "
              "shell scope=corpus or an immutable candidate_set_id; 5 total calls and 1024 observation tokens per call.")

    def validate(value):
        if set(value) != {"rubrics"}:
            raise RubricError("Initializer must return only rubrics")
        snapshot = validate_snapshot({**template, "rubrics": value["rubrics"],
            "origin": f"offline_model:{args.model};training_questions_sha256:{digest(questions)}"})
        if {r["id"] for r in snapshot["rubrics"]} != {r["id"] for r in template["rubrics"]} or any(r["revision"] for r in snapshot["rubrics"]):
            raise RubricError("Initializer must preserve the five ids at revision zero")
        return snapshot
    result = await client.complete("initialize", prompt, {"template": template, "training_questions": questions}, validate)
    atomic_json(args.output, result)
    atomic_json(str(args.output) + ".audit.json", {"origin": result["origin"], "training_qids": [q["qid"] for q in questions],
                "usage": client.usage, "snapshot_hash": digest(result), "policy_training_started": False})
    print(json.dumps({"initial_path": str(args.output), "snapshot_hash": digest(result), "policy_training_started": False}))


async def prepare_calibration(args):
    snapshot = load_snapshot(args.initial)
    records = read_jsonl(args.input)
    unique = {}
    for record in records:
        validate_events(record)
        if record.get("split") != "train" or not record.get("qid"):
            raise RubricError("Reference inputs must explicitly identify training-only qids")
        unique.setdefault(record["qid"], record)
    records = list(unique.values())[:args.question_limit]
    if len(records) < 8:
        raise RubricError("Prepare at least eight separate training reference questions")
    client = client_for(args)
    grades = await client.grade_many(snapshot, records)
    result = []
    for record, grade in zip(records, grades, strict=True):
        if grade is None:
            raise RubricError("Reference grading failed; do not publish a partial reference panel")
        result.append({**record, "reference_statuses": {k: g["status"] for k, g in grade.items()},
            "initial_snapshot_hash": digest(snapshot), "reference_source": f"model:{args.model};initial_rubric_judgment;not_human_labels",
            "reference_evidence": grade})
    path = Path(args.output)
    if path.exists():
        raise RubricError("Refusing to overwrite an existing reference panel")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf-8")
    print(json.dumps({"reference_questions": len(result), "output": str(path), "reference_type": "model_labeled_not_human", "policy_training_started": False}))


async def replay(args):
    snapshot = load_snapshot(args.initial)
    client = client_for(args)
    workflow = RubricWorkflow(snapshot, args.output, client, calibration_path=args.calibration)
    rows = read_jsonl(args.input)
    steps = {}
    for row in rows:
        if row.get("split") != "train":
            raise RubricError("Replay upkeep accepts only explicitly marked training trajectories")
        steps.setdefault(row["step"], []).append(row)
    report = []
    for step, records in sorted(steps.items()):
        workflow.begin_step(step)
        scored = await workflow.score_groups(records)
        retained = [r["trajectory_id"] for r in records if r.get("retained", True)]
        stats, decision = await workflow.finish_step(retained)
        report.append({"step": step, "scores": [r["score"] for r in scored], "statistics": stats, "decision": decision})
    atomic_json(Path(args.output) / "replay_report.json", {"steps": report, "policy_training_started": False})
    print(json.dumps({"steps_replayed": len(report), "output": str(args.output), "policy_training_started": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "initialize", "prepare-calibration", "replay", "export-inputs"):
        command = commands.add_parser(name)
        command.add_argument("--initial", default=str(INITIAL))
        command.add_argument("--base-url", default=os.environ.get("GREPSEEK_RUBRIC_URL", "http://127.0.0.1:8100/v1"))
        command.add_argument("--model", default=os.environ.get("GREPSEEK_RUBRIC_MODEL", "qwen38-27b-stop-judge"))
        command.add_argument("--concurrency", type=int, default=16)
        if name in {"check", "replay"}:
            command.add_argument("--calibration", default="")
        if name == "check":
            command.add_argument("--require-calibration", action="store_true")
            command.add_argument("--live", action="store_true")
        else:
            command.add_argument("--output", required=True)
        if name in {"prepare-calibration", "replay", "export-inputs"}:
            command.add_argument("--input", required=True)
        if name in {"initialize", "prepare-calibration"}:
            command.add_argument("--question-limit", type=int, default=64)
        if name in {"initialize", "export-inputs"}:
            command.add_argument("--train-file", required=True)
        if name == "export-inputs":
            command.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    try:
        if args.command == "check":
            check(args)
        elif args.command == "export-inputs":
            export_inputs(args)
        else:
            asyncio.run({"initialize": initialize, "prepare-calibration": prepare_calibration, "replay": replay}[args.command](args))
    except (RubricError, OSError, ValueError) as exc:
        parser.exit(2, f"rubric workflow: {exc}\n")
    except httpx.HTTPError:
        parser.exit(2, "rubric workflow: judge endpoint unavailable or returned an HTTP error\n")


if __name__ == "__main__":
    main()
