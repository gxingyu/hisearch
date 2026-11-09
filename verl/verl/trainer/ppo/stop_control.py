"""Verified, bounded auxiliary supervision of a stop header at the same prefix.

Qualification/probes are read-only. The resulting jobs are separate from all
outcome rewards, advantages, masks and GSPO importance ratios.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.answer_body_gate import AnswerBodyGate, HEADER, object_array

CONTENT_REGEX = r"[^< \t\r\n][^<]{0,255}</answer>"


def auxiliary_slots(count, world_size, minibatches, rank, minibatch):
    """Identical forward counts on every FSDP rank, without duplicating credit."""
    if min(world_size, minibatches) < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid auxiliary dispatch geometry")
    slots = []
    base = minibatch * world_size
    stride = world_size * minibatches
    while base < count:
        index = base + rank
        slots.append(index if index < count else None)
        base += stride
    return slots


class VerifiedStopControl:
    def __init__(self, tokenizer, config, *, model_length=16384, response_length=15360):
        self.tokenizer = tokenizer
        self.config = config
        self.max_qids = int(config.get("max_qids", 32))
        self.coefficient = float(config.get("coefficient", .05))
        self.seed = int(config.get("seed", 42))
        self.include_flat = bool(config.get("include_flat", True))
        self.model_length, self.response_length = model_length, response_length
        if not 1 <= self.max_qids <= 32 or not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("Invalid stop-control capacity/coefficient")
        self.scorer = AnswerBodyGate(tokenizer, p1_threshold=float(config.get("p1_threshold", .9)),
            relative_gain_threshold=float(config.get("relative_gain_threshold", .005)),
            max_relative_probability_drop=float(config.get("max_relative_probability_drop", .005)),
            max_model_len=model_length)
        self.header = tokenizer.encode(HEADER, add_special_tokens=False)
        if not self.header or tokenizer.decode(self.header, skip_special_tokens=False) != HEADER:
            raise ValueError("Stop header token round trip failed")

    def prepare(self, batch, score_fn, generate_fn, *, step, output_dir=None):
        """Run only on the rewarded FIRST generated batch, before filtering it."""
        from grepseek.eval.scoring import exact_match_score, max_f1_score
        from grepseek.trainer.verl_integration.reward_function import _is_valid_sequence

        if "retrieval_rounds" not in batch.non_tensor_batch or "rm_scores" not in batch.batch:
            raise RuntimeError("Stop control requires raw rewards and sampled round boundaries")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 0x570C]))
        uid_rows, qid_uids = defaultdict(list), {}
        for i, uid in enumerate(batch.non_tensor_batch["uid"]):
            uid_rows[str(uid)].append(i)
            qid = str(batch.non_tensor_batch["extra_info"][i]["qid"])
            qid_uids.setdefault(qid, str(uid))
        qids = list(qid_uids)
        selected = rng.choice(len(qids), size=min(self.max_qids, len(qids)), replace=False)
        scores = batch.batch["rm_scores"].sum(-1).detach().cpu().tolist()
        width = batch.batch["prompts"].shape[1]
        jobs, keys, records, trajectories = [], [], [], {}
        metrics = {"selected_qids": len(selected), "pool_rows": len(batch), "scored_prefixes": 0,
            "probability_eligible_rounds": 0, "candidate_qids": 0, "verified_qids": 0,
            "skipped_targets": 0, "skipped_prefixes": 0, "probe_choices": 0,
            "flat_verified_qids": 0, "auxiliary_examples": 0}
        for n in selected:
            qid = qids[int(n)]
            rows = uid_rows[qid_uids[qid]]
            if len(rows) != 5:
                raise ValueError("Verified stop control requires intact five-rollout groups")
            rewards = [scores[i] for i in rows]
            if not all(math.isfinite(x) for x in rewards):
                raise ValueError("Nonfinite first-batch reward")
            flat = max(rewards) == min(rewards)
            kind = ("flat_score1" if rewards[0] == 1 else "flat_score0" if rewards[0] == 0 else "flat_other") if flat else "active"
            record = {"qid": qid, "uid": qid_uids[qid], "source_rows": rows, "raw_rewards": rewards,
                      "flat": flat, "group_kind": kind, "comparisons": []}
            records.append(record)
            for i in rows:
                info = batch.non_tensor_batch["extra_info"][i]
                rounds = [dict(r) for r in batch.non_tensor_batch["retrieval_rounds"][i]]
                if len(rounds) < 2:
                    continue
                try:
                    header, body, answer = self.scorer.target(info["golden_answers"])
                    if header != self.header:
                        raise ValueError("header_tokenization_mismatch")
                except ValueError:
                    metrics["skipped_targets"] += 1
                    continue
                prompt = batch.batch["prompts"][i][batch.batch["attention_mask"][i, :width].bool()].tolist()
                previous_end = 0
                length = int(batch.batch["attention_mask"][i, width:].sum())
                prefixes = []
                for j, r in enumerate(rounds):
                    a, b, end = r["generated_start"], r["generated_end"], r["prefix_end"]
                    if r["round"] != j + 1 or a != previous_end or not 0 <= a < b < end <= length:
                        raise RuntimeError("Malformed sampled stop-control round")
                    if not bool(batch.batch["response_mask"][i, a:b].all()) or bool(batch.batch["response_mask"][i, b:end].any()):
                        raise RuntimeError("Stop-control round crosses observation/generated spans")
                    previous_end = end
                    prefix = prompt + batch.batch["responses"][i, :end].tolist()
                    prefixes.append(prefix)
                    ids = prefix + header + body[:-1]
                    if len(ids) > self.model_length:
                        metrics["skipped_prefixes"] += 1
                        continue
                    jobs.append((ids, body))
                    keys.append((i, j))
                trajectories[i] = {"prefixes": prefixes, "rounds": rounds, "gold": info["golden_answers"],
                    "fixed_gold": answer, "record": record}
        values = {}
        if jobs:
            request = DataProto.from_dict(tensors={"gate_job_id": torch.arange(len(jobs))}, non_tensors={
                "gate_input_ids": object_array([x for x, _ in jobs]),
                "gate_target_ids": object_array([y for _, y in jobs])})
            actual = score_fn(request).batch["answer_body_logp"].tolist()
            if len(actual) != len(keys):
                raise RuntimeError("Stop-control score count mismatch")
            for key, value in zip(keys, actual):
                if math.isfinite(value) and value <= 1e-8:
                    values[key] = value
                else:
                    metrics["skipped_prefixes"] += 1
            metrics["scored_prefixes"] = len(actual)
        candidates = defaultdict(list)
        for i, item in trajectories.items():
            for j in range(1, len(item["rounds"])):
                a, b = values.get((i, j - 1)), values.get((i, j))
                eligible = bool(a is not None and b is not None and a >= self.scorer.log_high
                    and self.scorer.log_min_ratio <= b - a <= self.scorer.log_gain)
                item["record"]["comparisons"].append({"source_row": i, "round_b": j + 1,
                    "logp1": a, "logp2": b, "probability_eligible": eligible,
                    "tool_name": item["rounds"][j]["tool_name"]})
                metrics["probability_eligible_rounds"] += int(eligible)
                if eligible:
                    prefix = item["prefixes"][j - 1]
                    response_tokens = item["rounds"][j - 1]["prefix_end"]
                    budget = min(int(self.config.get("probe_max_tokens", 256)),
                        self.model_length - len(prefix) - len(self.header),
                        self.response_length - response_tokens - len(self.header))
                    if budget > 0:
                        candidates[item["record"]["qid"]].append({"source_row": i, "prefix_index": j,
                            "round_b": j + 1, "prefix_ids": prefix, "budget": budget, "item": item})
        chosen, requests = [], []
        for qid, options in candidates.items():
            earliest = min(c["prefix_index"] for c in options)
            tied = [c for c in options if c["prefix_index"] == earliest]
            candidate = tied[int(rng.integers(len(tied)))]
            chosen.append(candidate)
            record = candidate["item"]["record"]
            record.update(source_row=candidate["source_row"], prefix_index=earliest,
                round_b=candidate["round_b"], prefix_ids=candidate["prefix_ids"],
                prefix_ids_sha256=hashlib.sha256(json.dumps(candidate["prefix_ids"]).encode()).hexdigest())
            for sample in range(2):
                seed = int(hashlib.sha256(f"{self.seed}:{step}:{qid}:{sample}".encode()).hexdigest()[:8], 16) % (2**31)
                requests.append({"request_id": f"stop-control-{step}-{seed}-{sample}",
                    "input_ids": candidate["prefix_ids"] + self.header,
                    "temperature": float(sample), "seed": seed, "max_tokens": candidate["budget"]})
        metrics["candidate_qids"] = len(chosen)
        outputs = generate_fn(requests) if requests else []
        if len(outputs) != len(requests):
            raise RuntimeError("Stop-control probe count mismatch")
        payload = []
        for n, candidate in enumerate(chosen):
            item, probes = candidate["item"], []
            record = item["record"]
            for sample in range(2):
                output = outputs[2*n + sample]
                ids = list(output["token_ids"])
                text = HEADER + self.tokenizer.decode(ids, skip_special_tokens=True)
                valid, reason = _is_valid_sequence("<think>\n" + text)
                answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
                prediction = answers[-1].strip() if answers else ""
                em = bool(prediction and exact_match_score(prediction, item["gold"]))
                f1 = max_f1_score(prediction, item["gold"]) if prediction else 0.
                probes.append({"sample": sample, "temperature": requests[2*n+sample]["temperature"],
                    "seed": requests[2*n+sample]["seed"], "token_ids": ids, "text": text,
                    "prediction": prediction, "format_pass": bool(valid), "format_reason": reason,
                    "em": em, "f1": f1, "passed": bool(valid and em and f1 >= .95),
                    "request_id": requests[2*n+sample]["request_id"]})
            verified = all(p["passed"] for p in probes)
            record.update(probes=probes, verified=verified, flat_excluded=bool(record["flat"] and not self.include_flat))
            metrics["probe_choices"] += len(probes)
            metrics["verified_qids"] += int(verified)
            metrics["flat_verified_qids"] += int(verified and record["flat"])
            if verified and (self.include_flat or not record["flat"]):
                payload.append({"qid": record["qid"], "uid": record["uid"],
                    "prefix_ids": candidate["prefix_ids"], "target_ids": self.header,
                    "prefix_index": candidate["prefix_index"], "source_row": candidate["source_row"],
                    "group_kind": record["group_kind"]})
                kind = "auxiliary_" + record["group_kind"]
                metrics[kind] = metrics.get(kind, 0) + 1
        metrics["auxiliary_examples"] = len(payload)
        metrics["nominal_mixture_weight"] = self.coefficient * len(payload) / 1280
        if output_dir is not None:
            folder = Path(output_dir) / "stop_control"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"{step}.jsonl").open("w") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return payload, {"stop_control/" + k: v for k, v in metrics.items()}
