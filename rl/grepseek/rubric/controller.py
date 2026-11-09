"""Driver-side adapter: raw F1 -> rubric reward -> existing GRPO/GSPO flow."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from grepseek.rubric.core import RubricError, RubricWorkflow, atomic_json, load_snapshot
from grepseek.rubric.client import RubricClient


def validate_training_configuration(config):
    algorithm = config.get("algorithm", {})
    if not algorithm.get("rubric", {}).get("enabled", False):
        return
    actor = config["actor_rollout_ref"]["actor"]
    if algorithm.get("adv_estimator") != "grpo" or actor.get("policy_loss", {}).get("loss_mode") != "gspo":
        raise RubricError("Five-rubric training requires the existing GRPO advantages and GSPO loss")
    if algorithm.get("use_kl_in_reward", False):
        raise RubricError("Rubric reward requires use_kl_in_reward=false")
    for name in ("stop_control", "answer_body_gate", "route_efficiency"):
        if algorithm.get(name, {}).get("enabled", False):
            raise RubricError(f"Disable {name} for F1 + five rubric rewards")
    if config.get("trainer", {}).get("critic_warmup", 0) != 0:
        raise RubricError("Rubric upkeep requires an actor update every completed outer step")
    reward = config.get("custom_reward_function") or config.get("reward", {}).get("custom_reward_function", {})
    kwargs = reward.get("reward_kwargs", {})
    if reward.get("name") != "grepseek_reward" or kwargs.get("reward_metric", "f1") != "f1" or kwargs.get("enable_length_decay", False):
        raise RubricError("Use grepseek_reward with raw F1 and length decay disabled")
    rubric = algorithm["rubric"]
    if rubric.get("per_item_reward", .1) != .1:
        raise RubricError("Each passed rubric must contribute exactly 0.1")


def object_array(values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


class RubricController:
    def __init__(self, config, *, client=None):
        self.config = dict(config)
        if self.config.get("per_item_reward", .1) != .1:
            raise RubricError("Rubric pass reward is fixed at 0.1")
        initial = load_snapshot(config["initial_path"])
        self.client = client or RubricClient(config["judge"])
        self.workflow = RubricWorkflow(initial, config["output_dir"], self.client,
            calibration_path=config.get("calibration_path", ""), maintenance=dict(config.get("maintenance", {})))

    def begin_step(self, step):
        self.usage_before = dict(getattr(self.client, "usage", {}))
        return self.workflow.begin_step(step)

    @staticmethod
    def _column(batch, name, default=None):
        return batch.non_tensor_batch.get(name, [default] * len(batch))

    def score_batch(self, batch):
        if "rubric/version" in batch.non_tensor_batch:
            if not all(int(v) == self.workflow.frozen["version"] for v in batch.non_tensor_batch["rubric/version"]):
                raise RubricError("Mixed rubric versions inside one outer step")
            return batch
        metadata = self._column(batch, "extra_info", {})
        events = self._column(batch, "rubric_events")
        identifiers = self._column(batch, "retrieval_request_id")
        format_pass = self._column(batch, "reward/format_pass")
        failed = self._column(batch, "reward/failed_rollout", 0)
        f1 = self._column(batch, "reward/f1")
        base = batch.batch["rm_scores"].sum(-1).detach().cpu().tolist()
        records = []
        for i, uid in enumerate(batch.non_tensor_batch["uid"]):
            info = metadata[i] or {}
            if info.get("is_validation", False):
                raise RubricError("Validation trajectories must never enter rubric upkeep")
            if format_pass[i] is None or f1[i] is None:
                raise RubricError("Base reward must expose format_pass and raw F1")
            valid = bool(format_pass[i]) and not bool(failed[i])
            if valid and events[i] is None:
                raise RubricError("Missing final-visible token ledger: enable the hybrid rubric agent path")
            qid = str(info.get("qid", ""))
            question = str(info.get("question", ""))
            if not qid or not question:
                raise RubricError("Rubric training needs qid and original question metadata")
            ledger = [{"id": "question", "kind": "question", "text": question}] + list(events[i] or [])
            trace = self._column(batch, "tool_trace", [])[i] or []
            buckets = set()
            for event in trace:
                scope = event.get("scope", event.get("tool_name", "unknown"))
                buckets.add(str(scope.get("type", "unknown") if isinstance(scope, dict) else scope))
                if event.get("truncated"):
                    buckets.add("truncated")
                if event.get("status") not in (None, "ok", "no_hits"):
                    buckets.add("execution_failure")
            records.append({"trajectory_id": str(identifiers[i] or f"{uid}:{i}"), "group_id": str(uid),
                "qid": qid, "question": question, "events": ledger, "valid": valid,
                "format_invalid": not bool(format_pass[i]), "failed_rollout": bool(failed[i]),
                "f1": float(f1[i]), "base_score": float(base[i]), "buckets": sorted(buckets)})
        scored = asyncio.run(self.workflow.score_groups(records))
        reward_tensor = batch.batch["rm_scores"].clone()
        for i, row in enumerate(scored):
            if row["bonus"]:
                positions = batch.batch["response_mask"][i].nonzero(as_tuple=False).flatten()
                if not len(positions):
                    raise RubricError("A valid rubric reward needs a generated response token")
                reward_tensor[i, int(positions[-1])] += row["bonus"]
        batch.batch["rm_scores"] = reward_tensor
        columns = {
            "score": [r["score"] for r in scored], "reward/total": [r["score"] for r in scored],
            "reward/rubric": [r["bonus"] for r in scored], "rubric/passed": [r["passed"] for r in scored],
            "rubric/fallback": [int(r["fallback"]) for r in scored],
            "rubric/masked_items": [r["masked_items"] for r in scored],
            "rubric/version": [r["rubric_version"] for r in scored],
        }
        for key, values in columns.items():
            batch.non_tensor_batch[key] = np.asarray(values)
        batch.non_tensor_batch["rubric_trajectory_id"] = object_array([r["trajectory_id"] for r in scored])
        keys = list(batch.meta_info.get("reward_extra_keys", []))
        batch.meta_info["reward_extra_keys"] = list(dict.fromkeys(keys + list(columns)))
        return batch

    def finish_step(self, retained_batch):
        stats, decision = asyncio.run(self.workflow.finish_step(retained_batch.non_tensor_batch["rubric_trajectory_id"].tolist()))
        metrics = {"rubric/all_sampled/trajectories": stats["trajectories"],
                   "rubric/all_sampled/question_groups": stats["question_groups"],
                   "rubric/all_sampled/distinct_questions": stats["distinct_questions"],
                   "rubric/all_sampled/judge_errors": stats["judge_errors"],
                   "rubric/all_sampled/format_invalid": stats["format_invalid"],
                   "rubric/all_sampled/failed_rollouts": stats["failed_rollouts"],
                   "rubric/retained/trajectories": stats["retained_trajectories"],
                   "rubric/retained/question_groups": stats["retained_question_groups"],
                   "rubric/maintenance/changed": int(decision["action"] != "KEEP"),
                   "rubric/next_version": self.workflow.active["version"]}
        for identifier, item in stats["items"].items():
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    metrics[f"rubric/items/{identifier}/{key}"] = value
        for key, value in getattr(self.client, "usage", {}).items():
            metrics[f"rubric/judge/{key}"] = value - self.usage_before.get(key, 0)
        return metrics

    def save_checkpoint(self, path):
        if self.workflow.step is not None:
            raise RubricError("Cannot checkpoint in the middle of a rubric scoring step")
        atomic_json(Path(path) / "rubric_state.json", self.workflow.state_dict())

    def abort_step(self, reason):
        self.workflow.abort_step(reason)

    def load_checkpoint(self, path, completed_step):
        state_path = Path(path) / "rubric_state.json"
        if not state_path.is_file():
            raise RubricError("Resume requires the rubric state saved with the policy checkpoint")
        self.workflow.restore(json.loads(state_path.read_text()), completed_step)
