"""Remove positive credit from stable, low-gain answer-body retrieval rounds."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from verl import DataProto

HEADER = "</think>\n<answer>\n"


def annealed_gate_weight(step, start_step=40, end_step=80):
    """Global-update schedule; a resumed run must use its restored step."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Gate step must be a nonnegative integer")
    z = min(1., max(0., (step - start_step) / (end_step - start_step)))
    return .5 * (1. - math.cos(math.pi * z))


def bounded_soft_gate(advantages, response_mask, selected, *, weight,
                      max_token_removal=.25, max_sequence_credit_removal=.10,
                      max_batch_credit_removal=.01):
    """Apply global, sequence-equally-weighted credit caps before annealing.

    This runs on the complete driver batch, never on separate DP minibatches.
    The caps constrain credit coefficients, not neural gradients or QA quality.
    """
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Soft gate weight must be in [0, 1]")
    if advantages.shape != response_mask.shape or selected.shape != advantages.shape:
        raise ValueError("Soft gate masks must match advantages")
    if not bool(torch.isfinite(advantages).all()) or bool((selected & ~response_mask.bool()).any()):
        raise ValueError("Nonfinite advantages or selected observation tokens")
    if any(not math.isfinite(x) or not 0 < x < 1 for x in
           (max_token_removal, max_sequence_credit_removal, max_batch_credit_removal)):
        raise ValueError("Soft gate credit caps must be strictly between zero and one")
    lengths = response_mask.sum(-1).double().clamp_min(1)
    positive = advantages.clamp_min(0) * response_mask
    positive_credit = positive.sum(-1, dtype=torch.float64) / lengths
    selected_credit = (positive * selected).sum(-1, dtype=torch.float64) / lengths
    alpha = torch.minimum(torch.full_like(lengths, max_token_removal),
                          max_sequence_credit_removal * positive_credit / (selected_credit + 1e-12))
    unscaled_removal = alpha * selected_credit
    kappa = min(1., max_batch_credit_removal * float(positive_credit.sum()) /
                (float(unscaled_removal.sum()) + 1e-12))
    scale = alpha * (weight * kappa)
    # Do not renormalize advantages, mask, rewards or the GSPO denominator.
    delta = positive * selected * scale.to(advantages.dtype).unsqueeze(-1)
    result = advantages - delta
    actual = (advantages.double() - result.double()).sum(-1) / lengths
    fraction = actual / positive_credit.clamp_min(1e-12)
    active = selected & (advantages > 0) & (delta > 0)
    metrics = {
        "annealing_weight": weight, "temperature": 1. - weight,
        "budget_kappa": kappa,
        "positive_sequence_credit_before": float(positive_credit.sum()),
        "positive_sequence_credit_removed": float(actual.sum()),
        "positive_credit_removed_fraction": float(actual.sum() / positive_credit.sum().clamp_min(1e-12)),
        "max_sequence_positive_credit_removed_fraction": float(fraction.max()),
        "max_effective_token_removal": float(scale.max()),
        "softened_generated_tokens": int(active.sum()),
        "fully_cleared_generated_tokens": int(((result == 0) & active).sum()),
    }
    return result, metrics, scale, positive_credit, actual


def object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class AnswerBodyGate:
    def __init__(self, tokenizer, *, p1_threshold=0.9, relative_gain_threshold=0.005,
                 max_relative_probability_drop=0.005, max_model_len=16384,
                 mode="hard", anneal_start_step=40, anneal_end_step=80,
                 max_token_removal=.25, max_sequence_credit_removal=.10,
                 max_batch_credit_removal=.01):
        if not 0 < p1_threshold <= 1 or not math.isfinite(relative_gain_threshold) or relative_gain_threshold < 0:
            raise ValueError("Invalid answer-body gate thresholds")
        if not math.isfinite(max_relative_probability_drop) or not 0 <= max_relative_probability_drop < 1:
            raise ValueError("Invalid maximum relative probability drop")
        self.tokenizer = tokenizer
        self.log_high = math.log(p1_threshold)
        self.log_gain = math.log1p(relative_gain_threshold)
        self.log_min_ratio = math.log(1.0 - max_relative_probability_drop)
        self.max_model_len = max_model_len
        if mode not in ("hard", "annealed_soft"):
            raise ValueError("Unknown answer-body gate mode")
        if (not isinstance(anneal_start_step, int) or not isinstance(anneal_end_step, int)
                or isinstance(anneal_start_step, bool) or isinstance(anneal_end_step, bool)
                or not 0 <= anneal_start_step < anneal_end_step):
            raise ValueError("Invalid gate annealing interval")
        if any(not math.isfinite(x) or not 0 < x < 1 for x in
               (max_token_removal, max_sequence_credit_removal, max_batch_credit_removal)):
            raise ValueError("Soft gate credit caps must be strictly between zero and one")
        self.mode = mode
        self.anneal_start_step, self.anneal_end_step = anneal_start_step, anneal_end_step
        self.max_token_removal = max_token_removal
        self.max_sequence_credit_removal = max_sequence_credit_removal
        self.max_batch_credit_removal = max_batch_credit_removal
        self.targets = {}

    def target(self, answers):
        # Use the fixed first nonempty alias, never maximize across aliases.
        answer = next((x for x in answers if isinstance(x, str) and x.strip()), None)
        if answer is None:
            raise ValueError("missing_gold")
        if answer not in self.targets:
            text = HEADER + answer + "\n</answer>"
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            start, end = len(HEADER), len(HEADER) + len(answer)
            indices = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
            if not indices or indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError("invalid_body_token_span")
            if not all(start <= enc["offset_mapping"][i][0] and enc["offset_mapping"][i][1] <= end for i in indices):
                raise ValueError("body_token_crosses_format_boundary")
            body = [enc["input_ids"][i] for i in indices]
            header = enc["input_ids"][:indices[0]]
            if self.tokenizer.decode(body, skip_special_tokens=False) != answer or self.tokenizer.decode(header, skip_special_tokens=False) != HEADER:
                raise ValueError("body_token_roundtrip_failed")
            self.targets[answer] = (header, body, answer)
        return self.targets[answer]

    def apply(self, batch: DataProto, score_fn, *, step: int, output_dir=None):
        if "retrieval_rounds" not in batch.non_tensor_batch:
            raise RuntimeError("Gate enabled but sampled retrieval-round token boundaries are absent")
        mask = batch.batch["response_mask"]
        advantages = batch.batch["advantages"]
        soft = self.mode == "annealed_soft"
        weight = annealed_gate_weight(step, self.anneal_start_step, self.anneal_end_step) if soft else 1.
        keep = torch.ones_like(mask, dtype=torch.bool)
        prompt_width = batch.batch["prompts"].shape[1]
        inputs, targets, job_keys, records = [], [], [], []
        prefix_scores = {}
        metrics = {"scored_prefixes": 0, "compared_rounds": 0, "gated_rounds": 0,
                   "skipped_prefixes": 0, "skipped_trajectories": 0,
                   "gated_shell_rounds": 0, "gated_dense_search_rounds": 0,
                   "probability_eligible_rounds": 0, "excluded_probability_drop_rounds": 0,
                   "preserved_negative_adv_tokens": 0}
        for i in range(len(batch)):
            rounds = batch.non_tensor_batch["retrieval_rounds"][i]
            if rounds is None:
                raise RuntimeError("Missing retrieval round metadata in an enabled batch")
            info = batch.non_tensor_batch["extra_info"][i]
            record = {"qid": info.get("qid"), "uid": str(batch.non_tensor_batch["uid"][i]),
                      "rounds": [dict(r) for r in rounds], "comparisons": []}
            records.append(record)
            # Validate every recorded span even if no adjacent comparison exists.
            previous_end = 0
            response_len = int(batch.batch["attention_mask"][i, prompt_width:].sum())
            for j, r in enumerate(rounds, 1):
                a, b, end = r["generated_start"], r["generated_end"], r["prefix_end"]
                if r["round"] != j or a != previous_end or not 0 <= a < b < end <= response_len:
                    raise RuntimeError(f"Malformed sampled retrieval round: {r}")
                if not bool(mask[i, a:b].all()) or bool(mask[i, b:end].any()):
                    raise RuntimeError("Retrieval span crosses model/observation token boundaries")
                previous_end = end
            if len(rounds) < 2:
                continue
            if soft and weight == 0:
                record["inactive_before_anneal_start"] = True
                continue
            try:
                header, body, answer = self.target(info["golden_answers"])
            except ValueError as exc:
                metrics["skipped_trajectories"] += 1
                record["skip_reason"] = str(exc)
                continue
            record["target_answer"] = answer
            record["target_tokens"] = len(body)
            prompt = batch.batch["prompts"][i][batch.batch["attention_mask"][i, :prompt_width].bool()].tolist()
            for j, r in enumerate(rounds):
                prefix = prompt + batch.batch["responses"][i, :r["prefix_end"]].tolist()
                ids = prefix + header + body[:-1]
                if len(ids) > self.max_model_len:
                    metrics["skipped_prefixes"] += 1
                    r = record["rounds"][j]
                    r["skip_reason"] = "probe_exceeds_context"
                    continue
                inputs.append(ids)
                targets.append(body)
                job_keys.append((i, j))
        if inputs:
            jobs = DataProto.from_dict(tensors={"gate_job_id": torch.arange(len(inputs))}, non_tensors={
                "gate_input_ids": object_array(inputs), "gate_target_ids": object_array(targets)})
            values = score_fn(jobs).batch["answer_body_logp"].tolist()
            if len(values) != len(job_keys):
                raise RuntimeError("Probe result count does not match dispatched prefixes")
            metrics["scored_prefixes"] = len(values)
            for key, value in zip(job_keys, values):
                if math.isfinite(value) and value <= 1e-8:
                    prefix_scores[key] = value
                    records[key[0]]["rounds"][key[1]]["body_logp"] = value
                else:
                    metrics["skipped_prefixes"] += 1
                    records[key[0]]["rounds"][key[1]]["skip_reason"] = "nonfinite_or_positive_logp"
        for i, record in enumerate(records):
            for j, r in enumerate(record["rounds"]):
                if j == 0:
                    continue
                p1, p2 = prefix_scores.get((i, j-1)), prefix_scores.get((i, j))
                if p1 is None or p2 is None:
                    record["comparisons"].append({"round": j+1, "gated": False, "skip_reason": "missing_valid_adjacent_score"})
                    continue
                delta = p2-p1
                excluded_drop = p1 >= self.log_high and delta < self.log_min_ratio
                eligible = p1 >= self.log_high and self.log_min_ratio <= delta <= self.log_gain
                a, b = r["generated_start"], r["generated_end"]
                positive = advantages[i, a:b] > 0
                positive_tokens = int(positive.sum()) if eligible else 0
                negative_tokens = int((advantages[i, a:b] < 0).sum()) if eligible else 0
                flagged = positive_tokens > 0
                metrics["compared_rounds"] += 1
                metrics["probability_eligible_rounds"] += int(eligible)
                metrics["excluded_probability_drop_rounds"] += int(excluded_drop)
                metrics["preserved_negative_adv_tokens"] += negative_tokens
                record["comparisons"].append({"round": j+1, "tool_name": r["tool_name"],
                    "logp1": p1, "logp2": p2, "p1": math.exp(p1), "p2": math.exp(p2),
                    "log_ratio": delta, "relative_gain": math.expm1(delta) if delta < 700 else None,
                    "probability_eligible": eligible, "excluded_probability_drop": excluded_drop,
                    "cleared_positive_tokens": 0 if soft else positive_tokens,
                    "eligible_positive_tokens": positive_tokens,
                    "softened_positive_tokens": positive_tokens if soft else 0,
                    "preserved_negative_tokens": negative_tokens,
                    "gated": flagged})
                if flagged:
                    keep[i, a:b] &= ~positive
                    metrics["gated_rounds"] += 1
                    tool_metric = f"gated_{r['tool_name']}_rounds"
                    metrics[tool_metric] = metrics.get(tool_metric, 0) + 1
        cleared = ~keep
        if bool((cleared & ~mask.bool()).any()):
            raise RuntimeError("Gate attempted to mask a non-model token")
        metrics["gated_generated_tokens"] = int(cleared.sum())
        metrics["generated_tokens"] = int(mask.sum())
        metrics["gated_positive_adv_tokens"] = int((cleared & (advantages > 0)).sum())
        metrics["gated_negative_adv_tokens"] = int((cleared & (advantages < 0)).sum())
        metrics["gated_zero_adv_tokens"] = int((cleared & (advantages == 0)).sum())
        metrics["gated_round_fraction"] = metrics["gated_rounds"] / max(1, metrics["compared_rounds"])
        metrics["gated_token_fraction"] = metrics["gated_generated_tokens"] / max(1, metrics["generated_tokens"])
        # Preserve all negative/zero credit, including harmful search reasoning.
        # Only advantages change. Rewards, returns, response_mask, ratios, KL and
        # all normalization/aggregation denominators retain their original values.
        if soft:
            result, soft_metrics, scale, positive_credit, removed_credit = bounded_soft_gate(
                advantages, mask, cleared, weight=weight,
                max_token_removal=self.max_token_removal,
                max_sequence_credit_removal=self.max_sequence_credit_removal,
                max_batch_credit_removal=self.max_batch_credit_removal)
            batch.batch["advantages"] = result
            metrics.update(soft_metrics)
            metrics.update({"anneal_start_step": self.anneal_start_step,
                            "anneal_end_step": self.anneal_end_step,
                            "max_token_removal": self.max_token_removal,
                            "max_sequence_credit_removal": self.max_sequence_credit_removal,
                            "max_batch_credit_removal": self.max_batch_credit_removal})
            for i, record in enumerate(records):
                record["soft_gate"] = {"weight": weight, "applied_token_fraction": float(scale[i]),
                    "positive_sequence_credit_before": float(positive_credit[i]),
                    "positive_sequence_credit_removed": float(removed_credit[i]),
                    "generated_tokens": int(mask[i].sum())}
                for comparison in record["comparisons"]:
                    if comparison.get("gated"):
                        comparison["applied_token_fraction"] = float(scale[i])
        else:
            batch.batch["advantages"] = advantages.masked_fill(cleared, 0)
        if output_dir is not None:
            directory = Path(output_dir) / "answer_body_gate"
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{step}.jsonl").open("w") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            (directory / f"{step}.summary.json").write_text(
                json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        return {f"answer_body_gate/{k}": v for k, v in metrics.items()}
