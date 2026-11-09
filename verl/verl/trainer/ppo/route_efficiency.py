"""No-branch efficiency learning within successful hybrid-tool skeletons.

Only the retained on-policy batch is used. Task rewards/advantages and dynamic
sampling are untouched. Finite-sample block balance is not a neural-policy
quality guarantee.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch


ADVANTAGE_KEY = "route_efficiency_advantages"
SCALE_KEY = "route_efficiency_length_scale"


def prepare_route_efficiency(batch, *, rollouts_per_prompt, max_calls=5, coefficient=.05,
                             step=None, output_dir=None):
    """Attach aligned per-trajectory auxiliary tensors; never alter task data."""
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Route-efficiency coefficient must be finite and nonnegative")
    if max_calls <= 0:
        raise ValueError("Route-efficiency max_calls must be positive")
    required = ("uid", "tool_counts", "tool_trace", "reward/em", "reward/f1",
                "reward/format_pass", "reward/invalid_tool_request", "metrics/natural_final_answer",
                "metrics/forced_final_answer")
    missing = [key for key in required if key not in batch.non_tensor_batch]
    if missing:
        raise ValueError(f"Route-efficiency metadata missing: {missing}")
    nt = batch.non_tensor_batch
    n = len(batch)
    lengths = batch.batch["response_mask"].sum(-1).detach().float()
    if n == 0 or not bool(torch.isfinite(lengths).all()) or bool((lengths < 0).any()) or lengths.sum() <= 0:
        raise ValueError("Route-efficiency requires a nonempty generated-token batch")
    reference_length = float(lengths.mean().item())
    task_scores = batch.batch["rm_scores"].sum(-1).detach().cpu().tolist()
    groups, buckets = defaultdict(list), defaultdict(list)
    advantages = np.zeros(n, dtype=np.float32)
    records = []
    for i in range(n):
        uid = nt["uid"][i]
        groups[uid].append(i)
        success = bool(nt["reward/format_pass"][i] and nt["reward/em"][i] == 1
                       and nt["reward/f1"][i] == 1 and task_scores[i] == 1
                       and not nt["reward/invalid_tool_request"][i]
                       and nt["metrics/natural_final_answer"][i]
                       and not nt["metrics/forced_final_answer"][i])
        counts = nt["tool_counts"][i] or {}
        cost = sum(int(counts.get(name, 0)) for name in ("dense_search", "shell"))
        trace = nt["tool_trace"][i] or []
        names = [event.get("tool_name") for event in trace]
        valid_trace = (all(name in ("dense_search", "shell") for name in names)
                       and Counter(names)["shell"] == counts.get("shell", 0)
                       and Counter(names)["dense_search"] == counts.get("dense_search", 0)
                       and 0 <= cost <= max_calls)
        if success and (not valid_trace or lengths[i] <= 0):
            raise ValueError(f"Successful route-efficiency trajectory has inconsistent executed tools: row {i}")
        skeleton = "".join("D" if name == "dense_search" else "H"
                           for j, name in enumerate(names) if j == 0 or name != names[j - 1])
        skeleton = (skeleton or "ANSWER") if valid_trace else "INVALID_TRACE"
        if success:
            buckets[uid, skeleton].append(i)
        info = (nt.get("extra_info", [None] * n)[i] or {})
        records.append({"step": step, "row": i, "uid": str(uid), "qid": info.get("qid"),
                        "task_score": float(task_scores[i]), "em": float(nt["reward/em"][i]),
                        "f1": float(nt["reward/f1"][i]), "format_pass": bool(nt["reward/format_pass"][i]),
                        "natural_final": bool(nt["metrics/natural_final_answer"][i]),
                        "forced_final": bool(nt["metrics/forced_final_answer"][i]),
                        "success": success, "skeleton": skeleton, "calls": cost,
                        "generated_tokens": int(lengths[i].item())})
    for uid, rows in groups.items():
        if len(rows) != rollouts_per_prompt:
            raise ValueError(f"Route-efficiency expected {rollouts_per_prompt} trajectories per uid, got {len(rows)}")
    active_buckets, active_groups, shell_groups = 0, set(), set()
    variance_sum = 0.
    for (uid, skeleton), rows in buckets.items():
        mean_cost = float(np.mean([records[i]["calls"] for i in rows]))
        values = [(mean_cost - records[i]["calls"]) / max_calls for i in rows]
        variance_sum += sum((records[i]["calls"] - mean_cost) ** 2 for i in rows)
        for i, value in zip(rows, values, strict=True):
            advantages[i] = value
            records[i]["bucket_size"] = len(rows)
            records[i]["bucket_mean_calls"] = mean_cost
        if any(value != 0 for value in values):
            active_buckets += 1
            active_groups.add(uid)
            if "H" in skeleton:
                shell_groups.add(uid)
    e = torch.as_tensor(advantages, device=lengths.device)
    batch.batch[ADVANTAGE_KEY] = e
    batch.batch[SCALE_KEY] = lengths / reference_length
    batch.meta_info["route_efficiency_coefficient"] = coefficient
    batch.meta_info["route_efficiency_reference_length"] = reference_length
    for i, record in enumerate(records):
        record["advantage"] = float(advantages[i])
        record["scaled_advantage"] = float(advantages[i]) * record["generated_tokens"] / reference_length
    metrics = {"route_efficiency/groups": len(groups), "route_efficiency/successful_paths": sum(r["success"] for r in records),
               "route_efficiency/eligible_groups": len(active_groups), "route_efficiency/eligible_buckets": active_buckets,
               "route_efficiency/eligible_group_fraction": len(active_groups) / len(groups),
               "route_efficiency/eligible_shell_groups": len(shell_groups),
               "route_efficiency/nonzero_paths": int(np.count_nonzero(advantages)),
               "route_efficiency/advantage_abs_mean": float(np.abs(advantages).mean()),
               "route_efficiency/advantage_abs_max": float(np.abs(advantages).max()),
               "route_efficiency/advantage_sum": float(advantages.astype(np.float64).sum()),
               "route_efficiency/reference_generated_length": reference_length,
               "route_efficiency/empirical_cost_descent_eta1": variance_sum / (n * max_calls),
               "route_efficiency/coefficient": coefficient}
    if output_dir is not None:
        directory = Path(output_dir) / "route_efficiency"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{step}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        (directory / f"{step}.summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def route_efficiency_loss(old_log_prob, log_prob, advantages, length_scale, response_mask, *, config,
                          rollout_is_weights=None):
    """Separately clipped GSPO auxiliary loss using the existing model forward.

    The global length compensation makes its initial score gradient proportional
    to E_i * grad log pi(tau_i), rather than E_i/L_i. No per-hit normalization.
    """
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo

    credit = (advantages.detach() * length_scale.detach()).unsqueeze(-1) * response_mask
    loss, metrics = compute_policy_loss_gspo(old_log_prob, log_prob, credit, response_mask,
                                            config=config, rollout_is_weights=rollout_is_weights)
    return loss, {f"actor/route_efficiency_{key}": value for key, value in metrics.items()}
