# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
"""Format-penalized EM/F1 reward with an optional length penalty."""

from __future__ import annotations

import math
import re
from typing import Any


_INVALID_TOOL_REQUEST_REASON_INDEX = {
    "": -1,
    "malformed_tool_call": 0,
    "wrong_tool_name": 1,
    "invalid_arguments": 2,
    "missing_command": 3,
    "non_string_command": 4,
    "empty_command": 5,
    "command_validation_error": 6,
    "tool_call_limit_exceeded": 7,
}


# ---------------------------------------------------------------------------
# Penalty-warmup step-file reader.
#
# The trainer (verl ray_trainer.fit loop) writes the current global_step to a
# small text file at the top of each training step. The reward function reads
# it here to scale the length-penalty coefficient.
#
# Cached for 1 second of wall-clock to avoid hammering the filesystem on every
# per-rollout reward call (~256×5 = 1280 calls per training step).
# ---------------------------------------------------------------------------
import os as _os
import time as _time

_STEP_CACHE: dict[str, tuple[float, int | None]] = {}
_STEP_CACHE_TTL_S = 1.0


def _read_global_step(step_file: str | None) -> int | None:
    """Read the current training step from a small text file the trainer writes.

    Returns the step as int, or None if the file is missing / unreadable.
    Cached for `_STEP_CACHE_TTL_S` seconds per file path.
    """
    if not step_file:
        return None
    now = _time.time()
    cached = _STEP_CACHE.get(step_file)
    if cached and (now - cached[0]) < _STEP_CACHE_TTL_S:
        return cached[1]
    step: int | None = None
    try:
        with open(step_file) as f:
            txt = f.read().strip()
        if txt:
            step = int(txt)
    except (OSError, ValueError):
        step = None
    _STEP_CACHE[step_file] = (now, step)
    return step


def _penalty_warmup_factor(
    *, enable: bool, warmup_steps: int, current_step: int | None
) -> float:
    """Compute the warmup multiplier in [0, 1] for the length-penalty
    coefficient.

    Behavior:
      - enable=False              → 1.0 (no warmup; penalty at full strength)
      - warmup_steps<=0           → 1.0 (warmup is a no-op)
      - current_step is None      → 1.0 (graceful fallback: if the trainer
                                   isn't writing the step file, behave as if
                                   warmup is off — full penalty. Caller should
                                   notice via the metrics/penalty_warmup_*
                                   logging that step couldn't be read.)
      - else                      → min(1, current_step / warmup_steps)
                                   (linear ramp 0 → 1 over the first
                                   warmup_steps training steps)
    """
    if not enable or warmup_steps <= 0:
        return 1.0
    if current_step is None:
        return 1.0
    return min(1.0, max(0.0, current_step / float(warmup_steps)))

# penalty_mode controls how p(L) is combined with the base reward:
#   "additive"          score = base - p(L)            (default; preserves length
#                                                       variation on wrong rollouts)
#   "multiplicative"    score = base * (1 - p(L))      (legacy form; collapses to
#                                                       0 whenever base = 0)
#   "additive_clamped"  score = max(0, base - p(L))    (additive but clamped at
#                                                       0; removes any incentive
#                                                       for the model to "give
#                                                       up short" — wrong+short
#                                                       and wrong+long both
#                                                       score 0. Note: this
#                                                       re-flattens all-wrong
#                                                       GRPO groups, the thing
#                                                       additive was designed to
#                                                       break. Pick this if you
#                                                       observe the model gaming
#                                                       the length penalty by
#                                                       producing short garbage.)
_PENALTY_MODE_INDEX = {"additive": 0, "multiplicative": 1, "additive_clamped": 2}


def _compute_length_penalty(L: int, *, decay_type: str, a: float, L_max: int) -> float:
    """Compute the **additive** length penalty p(L) ∈ [0, a].

    p(0) = 0  (no penalty for a short response)
    p(L_max) = a  (full penalty at the response budget)

    Returned value is SUBTRACTED from the base reward — see usage at the bottom
    of `grepseek_reward`. `a` parameter is the *maximum* penalty,
    typically 0.1–0.3 (much smaller than the multiplicative-floor sense it had
    in the prior `_compute_decay` formulation, which mapped a→f(L_max)).

    `decay_type` shapes how the penalty grows over [0, L_max]:
        linear        p(t) = a * t
        quadratic     p(t) = a * t^2          (mild for short, harsh for long)
        cosine        p(t) = a * 0.5*(1-cos(πt))  (smooth S-curve)
        exponential   p(t) = a * (exp(t)-1)/(e-1) (slow start, sharp end)
    """
    t = min(L, L_max) / max(L_max, 1)  # normalized position in [0, 1]
    if decay_type == "linear":
        return a * t
    if decay_type == "quadratic":
        return a * (t**2)
    if decay_type == "exponential":
        return a * (math.exp(t) - 1.0) / (math.e - 1.0)
    if decay_type == "cosine":
        return a * 0.5 * (1.0 - math.cos(math.pi * t))
    raise ValueError(f"Unknown decay_type: {decay_type!r}. Choose from: linear, exponential, quadratic, cosine")


def _extract_prediction(solution_str: str) -> str:
    """Extract the model's final plain-text answer from a multi-turn trajectory string.

    The SFT'd model is trained to wrap its final answer in <answer>...</answer>
    tags. Take the contents of the last such block as the prediction.

    Falls back to legacy behavior (strip tool/think blocks, take last non-empty
    line) when no <answer> block is present, so older RL prompts that asked for
    plain-text-only finals still work.
    """
    answer_blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", solution_str, flags=re.DOTALL)
    if answer_blocks:
        return answer_blocks[-1].strip()

    clean = re.sub(r"<tool_call>.*?</tool_call>", "", solution_str, flags=re.DOTALL)
    clean = re.sub(r"<tool_response>.*?</tool_response>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ─────────────────────────────────────────────────────────────────────────────
# Format validation of the rollout transcript.
# Verified: 200/200 cold-start SFT trajectories pass this check, so trajectories
# that follow the SFT-learned format avoid the format penalty.
# ─────────────────────────────────────────────────────────────────────────────
_FORMAT_TAGS = ("reason", "think", "tool_call", "tool_response", "answer")
_SPLIT_PATTERN = re.compile(r"(</?(?:reason|think|tool_call|tool_response|answer)>)")
_TAG_PATTERN = re.compile(r"</?(?:reason|think|tool_call|tool_response|answer)>")
_ROLE_MARKERS = {"user", "assistant"}
_CHINESE_CHAR_PATTERN = re.compile(r"[一-鿿]")


def _is_valid_sequence(text: str) -> tuple[bool, str]:
    """Strict structural check on a rollout transcript.

    Enforces correct nesting of <think>/<reason>, <tool_call>, <tool_response>,
    <answer> blocks and that the trajectory terminates inside an <answer>.
    """
    text = text.strip()
    for tag in _FORMAT_TAGS:
        if len(re.findall(rf"<{tag}>", text)) != len(re.findall(rf"</{tag}>", text)):
            return False, f"Tag mismatch for {tag}"
    parts = _SPLIT_PATTERN.split(text)
    state = "start"
    for part in parts:
        clean_part = part.strip()
        if not clean_part:
            continue
        is_tag = _TAG_PATTERN.match(clean_part)
        if is_tag:
            if state == "end":
                return False, f"Unexpected tag '{clean_part}' after sequence completion"
            if clean_part in ("<reason>", "<think>"):
                if state in ("start", "after_tool_response"):
                    state = "in_reason"
                else:
                    return False, f"Unexpected {clean_part} at state {state}"
            elif clean_part in ("</reason>", "</think>"):
                if state != "in_reason":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "after_reason"
            elif clean_part == "<tool_call>":
                if state != "after_reason":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "in_tool_call"
            elif clean_part == "</tool_call>":
                if state != "in_tool_call":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "after_tool_call"
            elif clean_part == "<tool_response>":
                if state != "after_tool_call":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "in_tool_response"
            elif clean_part == "</tool_response>":
                if state != "in_tool_response":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "after_tool_response"
            elif clean_part == "<answer>":
                if state != "after_reason":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "in_answer"
            elif clean_part == "</answer>":
                if state != "in_answer":
                    return False, f"Unexpected {clean_part} at state {state}"
                state = "end"
        else:
            if state == "end":
                return False, f"Invalid trailing content after </answer>: '{clean_part[:20]}...'"
            if state in ("start", "after_reason", "after_tool_call", "after_tool_response"):
                if clean_part.lower() in _ROLE_MARKERS:
                    continue
                return False, f"Unexpected text '{clean_part[:15]}' outside tags at state {state}"
            if state in ("in_reason", "in_tool_call", "in_tool_response", "in_answer"):
                content_only = "".join(clean_part.split())
                if content_only:
                    cn_count = len(_CHINESE_CHAR_PATTERN.findall(content_only))
                    if cn_count / len(content_only) > 0.5:
                        return False, f"Chinese density exceeded in {state}"
    if state == "end":
        return True, "Valid sequence"
    return False, f"Incomplete: ended in {state}"


async def grepseek_stop_judge_reward(
    data_source, solution_str, ground_truth, extra_info, *, stop_judge_enabled=False,
    stop_judge_model="glm-5-3-flash", stop_judge_base_url="http://127.0.0.1:8100/v1",
    stop_judge_api_key_env="GREPSEEK_STOP_JUDGE_API_KEY", stop_judge_penalty=0.2,
    stop_judge_concurrency=2, stop_judge_timeout=60, stop_judge_retries=2,
    stop_judge_max_tokens=2048, stop_judge_cache_dir="", stop_judge_audit_path="",
    stop_judge_disable_thinking=True, stop_judge_thinking_parameter="enable_thinking", **kwargs,
):
    from grepseek.trainer.verl_integration.stop_judge import JudgeConfig, JudgeError, get_judge, parse_prefixes

    result = grepseek_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)
    result.update({
        "reward/before_stop_penalty": result["score"], "reward/stop_penalty": 0.0,
        "reward/judge": 0.0,
        "judge/enabled": int(stop_judge_enabled), "judge/evaluated": 0, "judge/triggered": 0,
        "judge/first_ready_tool_index": -1, "judge/extra_tool_calls": 0, "judge/skip_reason": 0,
        "judge/error": 0, "judge/api_calls": 0, "judge/cache_hits": 0,
        "judge/input_tokens": 0, "judge/output_tokens": 0, "judge/reasoning_tokens": 0,
        "judge/cached_tokens": 0, "judge/retries": 0, "judge/latency": 0.0,
        "judge/quote_mismatches": 0, "judge/uncertain_prefixes": 0,
    })
    if not stop_judge_enabled or extra_info.get("is_validation", False):
        result["judge/skip_reason"] = 1 if not stop_judge_enabled else 2
        return result
    if not result["reward/format_pass"] or result["reward/failed_rollout"] or result["metrics/is_no_final_answer"]:
        result["judge/skip_reason"] = 3
        return result
    if stop_judge_penalty != 0.2:
        raise ValueError("The stop-judge v1 penalty must be 0.2 per extra tool call")
    try:
        prefixes = parse_prefixes(solution_str)
    except JudgeError:
        result["judge/skip_reason"] = 4
        return result
    if not prefixes:
        result["judge/skip_reason"] = 5
        return result
    question = extra_info.get("question")
    if not isinstance(question, str) or not question.strip():
        raise JudgeError("Missing original question in reward extra_info")
    config = JudgeConfig(
        model=stop_judge_model, base_url=stop_judge_base_url, api_key_env=stop_judge_api_key_env,
        concurrency=stop_judge_concurrency, timeout=stop_judge_timeout, retries=stop_judge_retries,
        max_tokens=stop_judge_max_tokens, cache_dir=stop_judge_cache_dir, audit_path=stop_judge_audit_path,
        disable_thinking=stop_judge_disable_thinking,
        thinking_parameter=stop_judge_thinking_parameter,
    )
    metrics = await get_judge(config).score(question, ground_truth.get("golden_answers", []), prefixes)
    result["judge/evaluated"] = 1
    result["judge/triggered"] = int(metrics["first_ready_tool_index"] != -1)
    for field, value in metrics.items():
        result["judge/" + field] = value
    penalty = stop_judge_penalty * metrics["extra_tool_calls"]
    result["reward/stop_penalty"] = penalty
    result["reward/judge"] = -penalty
    result["score"] -= penalty
    result["reward/total"] = result["score"]
    return result


def grepseek_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    *,
    decay_type: str = "linear",
    a: float = 0.5,
    L_max: int = 12288,
    failed_rollout_reward: float = 0.0,
    format_penalty: float = 1.0,
    enable_length_decay: bool = True,
    reward_metric: str = "em",
    penalty_mode: str = "additive",
    # Penalty-warmup kwargs (see _penalty_warmup_factor docstring):
    enable_penalty_warmup: bool = False,
    penalty_warmup_steps: int = 0,
    current_step_file: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute format-penalized BASE − α·p(L) reward for a GrepSeek trajectory.

    BASE is selected by `reward_metric`:
      - "em" (default): binary {0, 1} exact match against gold answers
      - "f1": continuous [0, 1] token-overlap F1 against gold answers
    Both EM and F1 are computed for valid trajectories and logged as reward/em
    and reward/f1 regardless of which one drives training.

    Reward composition (in order):
      1. If the tool request or transcript format is invalid → score = -format_penalty
      2. Else if rollout is failed                → score = failed_rollout_reward
      3. Else: combine BASE and p(L) according to `penalty_mode`:
           additive          score = BASE − p(L)
           additive_clamped  score = max(0, BASE − p(L))
           multiplicative    score = BASE × (1 − p(L))
         where p(L) ∈ [0, a] grows from 0 (short response) to `a` (at L_max)
         along `decay_type` shape. When enable_length_decay=False, p(L)=0.

    Why three modes:
      - additive (default) keeps a length signal on wrong rollouts so all-wrong
        GRPO groups aren't flat. Tradeoff: the model can "give up short" to
        avoid the penalty (wrong+short scores 0, wrong+long scores -a).
      - additive_clamped is the same as additive for correct rollouts but pins
        wrong rollouts to 0 regardless of length. Closes the "give up short"
        gaming at the cost of re-flattening all-wrong groups.
      - multiplicative is the legacy form (score = 0 on every wrong rollout).

    Args:
        data_source: Question ID (qid).
        solution_str: Decoded full response text from VERL rollout.
        ground_truth: Dict with "golden_answers" list from the dataset.
        extra_info: Agent loop extra_fields; contains "response_token_count" and
            optionally "hit_max_context" (set by GrepSeekAgentLoop).
        decay_type: Shape of p(L) over [0, L_max] — "linear" | "quadratic"
            | "cosine" | "exponential".
        a: MAX penalty subtracted at L=L_max. p(0)=0, p(L_max)=a. Typical
            values 0.1–0.3 (much smaller than the prior multiplicative-floor
            sense, where a meant "floor at L_max" and was usually 0.5–0.75).
        L_max: Token count at which the penalty saturates. Should equal
            max_response_length.
        failed_rollout_reward: Reward assigned when no final answer is produced.
        format_penalty: Positive magnitude subtracted for invalid training trajectories.
        enable_length_decay: Master switch. If False, p(L)=0 (no length term).

    Returns:
        Dict with "score" (primary training signal) plus logging metrics.
    """
    from grepseek.eval.scoring import exact_match_score, max_f1_score  # local import for Ray worker safety

    L = int(extra_info.get("response_token_count", 0))
    hit_max_context = bool(extra_info.get("hit_max_context", False))
    invalid_tool_request = bool(extra_info.get("invalid_tool_request", False))
    invalid_tool_call_count = int(extra_info.get("invalid_tool_call_count", 0) or 0)
    invalid_tool_request_reason = str(extra_info.get("invalid_tool_request_reason", ""))
    invalid_tool_request_reason_index = _INVALID_TOOL_REQUEST_REASON_INDEX.get(invalid_tool_request_reason, -1)
    executed_calls = sum((extra_info.get("tool_counts") or {}).values())
    tool_limit = int(extra_info.get("tool_call_limit", 0) or 0)
    forced_answer = bool(extra_info.get("forced_final_answer", False))
    rollout_metrics = {
        "metrics/truncate_invalid_mode": float(extra_info.get("tool_limit_behavior") == "truncate_invalid"),
        "metrics/tool_call_limit_exceeded": float(invalid_tool_request_reason == "tool_call_limit_exceeded"),
        "metrics/tool_call_attempt_count": int(extra_info.get("tool_call_attempt_count", 0) or 0),
        "metrics/generated_token_count": int(extra_info.get("generated_token_count", 0) or 0),
        "metrics/forced_final_answer": float(forced_answer),
        "metrics/tool_call_limit_reached": float(bool(tool_limit and executed_calls >= tool_limit)),
        "metrics/natural_final_answer": 0.0,
        "metrics/natural_final_at_tool_limit": 0.0,
    }

    # During validation we always report raw EM (no length penalty), regardless
    # of the training-time `enable_length_decay` setting. The dataset marks val
    # items via extra_info["is_validation"]=True (see GrepSeekDataset.__getitem__).
    is_validation = bool(extra_info.get("is_validation", False))
    if is_validation:
        enable_length_decay = False

    # ── Penalty warmup ──────────────────────────────────────────────────────
    # Read current global_step from a file the trainer writes at every step.
    # Compute `a_effective = a * warmup_factor` where warmup_factor in [0, 1]
    # ramps linearly from 0 → 1 over the first `penalty_warmup_steps` steps.
    # When enable_penalty_warmup=False or the file can't be read, warmup_factor
    # falls back to 1.0 (penalty at full strength — current behavior).
    #
    # During validation, warmup is irrelevant (no penalty anyway).
    _current_step = _read_global_step(current_step_file) if (
        enable_penalty_warmup and not is_validation
    ) else None
    _warmup_factor = _penalty_warmup_factor(
        enable=enable_penalty_warmup and not is_validation,
        warmup_steps=penalty_warmup_steps,
        current_step=_current_step,
    )
    a_effective = a * _warmup_factor

    format_input = solution_str
    if not format_input.lstrip().startswith("<think>") and not format_input.lstrip().startswith("<reason>"):
        format_input = "<think>\n" + format_input
    format_pass, _ = _is_valid_sequence(format_input)

    if invalid_tool_request or not format_pass:
        penalty = 0.0 if is_validation else -format_penalty
        is_empty_response = len(solution_str.strip()) == 0
        return {
            **rollout_metrics,
            "score": penalty,
            "reward/total": penalty,
            "reward/em": 0.0,
            "reward/f1": 0.0,
            "reward/base": 0.0,
            "reward/length_decay": 0.0,
            "reward/length_penalty": 0.0,
            "reward/format_penalty": penalty,
            "reward/format_pass": 0.0,
            "reward/failed_rollout": float(is_empty_response),
            "reward/hit_max_context": float(hit_max_context),
            "reward/invalid_tool_request": float(invalid_tool_request),
            "metrics/response_token_count": L,
            "metrics/is_empty_response": float(is_empty_response),
            "metrics/is_no_final_answer": 0.0,
            "metrics/invalid_tool_call_count": invalid_tool_call_count,
            "metrics/invalid_tool_request_reason_index": invalid_tool_request_reason_index,
            "metrics/prediction_len_chars": 0,
            "metrics/decay_type_index": {"linear": 0, "exponential": 1, "quadratic": 2, "cosine": 3}.get(decay_type, -1),
            "metrics/a": a,
            "metrics/L_max": L_max,
            "metrics/length_decay_enabled": float(enable_length_decay),
            "metrics/penalty_mode_index": _PENALTY_MODE_INDEX.get((penalty_mode or "additive").lower().strip(), -1),
            "metrics/penalty_warmup_enabled": float(enable_penalty_warmup),
            "metrics/penalty_warmup_steps": penalty_warmup_steps,
            "metrics/warmup_factor": _warmup_factor,
            "metrics/effective_a": a_effective,
            "metrics/current_step_seen": _current_step if _current_step is not None else -1,
        }

    # ── Detect failed rollouts ──────────────────────────────────────────────
    # Case 1: zero tokens generated (VERL-level abort / immediate EOS)
    is_empty_response = len(solution_str.strip()) == 0

    # Case 2: model never produced a plain-text final answer — it only made
    # tool calls and was then cut off (hit max context or max turns) without
    # outputting anything after the last </tool_call>.
    prediction = _extract_prediction(solution_str)
    no_final_answer = (not is_empty_response) and prediction == ""

    is_failed = is_empty_response or no_final_answer

    if is_failed:
        score = failed_rollout_reward
        return {
            **rollout_metrics,
            "score": score,
            "reward/total": score,
            "reward/em": 0.0,
            "reward/f1": 0.0,
            "reward/base": 0.0,
            "reward/length_decay": 0.0,
            "reward/length_penalty": 0.0,
            "reward/format_penalty": 0.0,
            "reward/format_pass": 1.0,
            "reward/failed_rollout": 1.0,
            "reward/hit_max_context": float(hit_max_context),
            "reward/invalid_tool_request": 0.0,
            "metrics/response_token_count": L,
            "metrics/is_empty_response": float(is_empty_response),
            "metrics/is_no_final_answer": float(no_final_answer),
            "metrics/invalid_tool_call_count": invalid_tool_call_count,
            "metrics/invalid_tool_request_reason_index": invalid_tool_request_reason_index,
            "metrics/prediction_len_chars": len(prediction),
            "metrics/decay_type_index": {"linear": 0, "exponential": 1, "quadratic": 2, "cosine": 3}.get(decay_type, -1),
            "metrics/a": a,
            "metrics/L_max": L_max,
            "metrics/length_decay_enabled": float(enable_length_decay),
            "metrics/penalty_mode_index": _PENALTY_MODE_INDEX.get((penalty_mode or "additive").lower().strip(), -1),
            "metrics/penalty_warmup_enabled": float(enable_penalty_warmup),
            "metrics/penalty_warmup_steps": penalty_warmup_steps,
            "metrics/warmup_factor": _warmup_factor,
            "metrics/effective_a": a_effective,
            "metrics/current_step_seen": _current_step if _current_step is not None else -1,
        }

    # ── base − α·p(L) reward ──────────────────────────────────────────────────
    # ADDITIVE: penalty is subtracted (not multiplied) so wrong-answer rollouts
    # also see length variation in their reward, breaking GRPO group flatness
    # on the all-wrong / all-correct cases.
    #
    # We compute BOTH EM and F1 every step so they're always available for
    # logging; `reward_metric` selects which one drives the training signal.
    #   reward_metric="em"  → score = EM − p(L)        (binary {0,1} base)
    #   reward_metric="f1"  → score = F1 − p(L)        (continuous [0,1] base)
    golden_answers: list[str] = ground_truth.get("golden_answers", [])
    em = 1.0 if (golden_answers and exact_match_score(prediction, golden_answers)) else 0.0
    f1 = max_f1_score(prediction, golden_answers) if golden_answers else 0.0

    _metric = (reward_metric or "em").lower().strip()
    if _metric == "f1":
        base = f1
    elif _metric == "em":
        base = em
    else:
        raise ValueError(f"reward_metric must be 'em' or 'f1', got {reward_metric!r}")

    length_penalty = (_compute_length_penalty(L, decay_type=decay_type, a=a_effective, L_max=L_max)
                      if enable_length_decay else 0.0)

    # `penalty_mode` selects how p(L) combines with the base reward.
    #   "additive"          score = base - p(L)
    #                       Default. Keeps a length gradient on wrong rollouts.
    #   "multiplicative"    score = base * (1 - p(L))
    #                       Legacy; zero whenever base = 0 (collapses all-wrong
    #                       GRPO groups).
    #   "additive_clamped"  score = max(0, base - p(L))
    #                       Length signal only differentiates among correct
    #                       rollouts; wrong+short and wrong+long both score 0.
    #                       Removes "give up short to game the penalty" — at
    #                       the cost of re-flattening all-wrong groups.
    _mode = (penalty_mode or "additive").lower().strip()
    if _mode == "additive":
        raw_score = base - length_penalty
    elif _mode == "additive_clamped":
        raw_score = max(0.0, base - length_penalty)
    elif _mode == "multiplicative":
        raw_score = base * (1.0 - length_penalty)
    else:
        raise ValueError(
            "penalty_mode must be 'additive' | 'additive_clamped' | "
            f"'multiplicative', got {penalty_mode!r}"
        )
    score = raw_score

    return {
        **rollout_metrics,
        "metrics/natural_final_answer": float(not forced_answer),
        "metrics/natural_final_at_tool_limit": float(
            not forced_answer and bool(tool_limit and executed_calls >= tool_limit)
        ),
        "score": score,
        "reward/total": score,
        "reward/em": em,
        "reward/f1": f1,
        "reward/base": base,            # whichever metric is driving training (em or f1)
        # Penalty value (>=0, subtracted from base). Keep old key name so wandb
        # dashboards keep working, but semantics changed: now it's the penalty,
        # not the multiplier. Add explicit penalty key for new dashboards.
        "reward/length_decay": length_penalty,
        "reward/length_penalty": length_penalty,
        "reward/format_penalty": 0.0,
        "reward/format_pass": float(format_pass),
        "reward/failed_rollout": 0.0,
        "reward/hit_max_context": float(hit_max_context),
        "reward/invalid_tool_request": 0.0,
        "metrics/response_token_count": L,
        "metrics/is_empty_response": 0.0,
        "metrics/is_no_final_answer": 0.0,
        "metrics/invalid_tool_call_count": invalid_tool_call_count,
        "metrics/invalid_tool_request_reason_index": invalid_tool_request_reason_index,
        "metrics/prediction_len_chars": len(prediction),
        "metrics/decay_type_index": {"linear": 0, "exponential": 1, "quadratic": 2, "cosine": 3}.get(decay_type, -1),
        "metrics/a": a,
        "metrics/L_max": L_max,
        "metrics/length_decay_enabled": float(enable_length_decay),
        "metrics/penalty_mode_index": _PENALTY_MODE_INDEX.get(_mode, -1),
        "metrics/penalty_warmup_enabled": float(enable_penalty_warmup),
        "metrics/penalty_warmup_steps": penalty_warmup_steps,
        "metrics/warmup_factor": _warmup_factor,
        "metrics/effective_a": a_effective,
        "metrics/current_step_seen": _current_step if _current_step is not None else -1,
    }


def _has_no_final_answer(solution_str: str, hit_max_context: bool) -> bool:
    if re.search(r"<answer>\s*\S.*?</answer>", solution_str, flags=re.DOTALL):
        return False
    if not solution_str.strip() or hit_max_context or "<answer>" in solution_str:
        return True
    if solution_str.rfind("<think>") > solution_str.rfind("</think>"):
        return True
    return solution_str.rstrip().endswith(("</think>", "</tool_call>", "</tool_response>"))


def grepseek_efficiency_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    *,
    call_cost_rate: float = 0.02,
    free_tool_calls: int = 1,
    generation_free_tokens: int = 512,
    generation_buffer_tokens: int = 1536,
    generation_penalty_factor: float = 0.2,
    failed_rollout_reward: float = -1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """F1-weighted call cost plus a buffer penalty on sampled model tokens.

    Normal completed rollouts receive
        F1 * (1 - call_cost_rate * max(N - free_tool_calls, 0)) - P_generated.
    The first call is free. Empty/error/no-op returns still consume a call.
    Validation uses the original unshaped reward and EM/F1 metrics.
    """
    if not math.isfinite(call_cost_rate) or not 0 <= call_cost_rate <= 1:
        raise ValueError("call_cost_rate must be finite and in [0, 1]")
    if free_tool_calls < 0 or generation_free_tokens < 0 or generation_buffer_tokens <= 0:
        raise ValueError("Invalid free-call or generation-buffer thresholds")
    if not math.isfinite(generation_penalty_factor) or generation_penalty_factor < 0:
        raise ValueError("generation_penalty_factor must be finite and nonnegative")
    for key in ("tool_call_attempt_count", "generated_token_count"):
        value = extra_info.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer from the agent loop")
    calls = extra_info["tool_call_attempt_count"]
    generated = extra_info["generated_token_count"]
    if generated > int(extra_info.get("response_token_count", generated)):
        raise ValueError("generated_token_count exceeds response_token_count")
    if kwargs.get("reward_metric", "f1") != "f1":
        raise ValueError("grepseek_efficiency_reward requires reward_metric=f1")

    options = dict(kwargs)
    options.update(
        reward_metric="f1",
        enable_length_decay=False,
        enable_penalty_warmup=False,
        failed_rollout_reward=0.0 if extra_info.get("is_validation") else failed_rollout_reward,
    )
    result = grepseek_reward(data_source, solution_str, ground_truth, extra_info, **options)
    result.update({
        "metrics/tool_call_attempt_count": calls,
        "metrics/generated_token_count": generated,
        "metrics/non_generated_response_token_count": int(extra_info.get("response_token_count", generated)) - generated,
        "metrics/forced_final_answer": float(bool(extra_info.get("forced_final_answer"))),
        "metrics/call_cost_rate": call_cost_rate,
        "metrics/free_tool_calls": free_tool_calls,
        "metrics/generation_free_tokens": generation_free_tokens,
        "metrics/generation_buffer_tokens": generation_buffer_tokens,
        "metrics/generation_penalty_factor": generation_penalty_factor,
        "reward/call_penalty": 0.0,
        "reward/generation_penalty": 0.0,
        "reward/call_discount": 1.0,
    })
    # Preserve every original validation metric, including format-error handling.
    if extra_info.get("is_validation"):
        return result
    if not result["reward/format_pass"] or result["reward/failed_rollout"]:
        return result

    call_penalty = result["reward/f1"] * call_cost_rate * max(calls - free_tool_calls, 0)
    generation_penalty = generation_penalty_factor * min(
        1.0, max(0.0, (generated - generation_free_tokens) / generation_buffer_tokens)
    )
    score = result["reward/f1"] - call_penalty - generation_penalty
    result.update({
        "score": score,
        "reward/total": score,
        "reward/call_penalty": call_penalty,
        "reward/generation_penalty": generation_penalty,
        "reward/call_discount": 1.0 - call_cost_rate * max(calls - free_tool_calls, 0),
        "reward/length_decay": generation_penalty,
        "reward/length_penalty": generation_penalty,
    })
    return result


def grepseek_turn_think_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    *,
    think_threshold: int = 512,
    think_penalty_weight: float = 0.1,
    failed_rollout_reward: float = -0.2,
    **kwargs: Any,
) -> dict[str, Any]:
    if think_threshold <= 0:
        raise ValueError("think_threshold must be positive")
    if think_penalty_weight < 0:
        raise ValueError("think_penalty_weight must be nonnegative")
    if "turn_think_token_counts" not in extra_info:
        raise ValueError("turn_think_token_counts missing from agent loop output")

    turn_lengths = [int(length) for length in extra_info["turn_think_token_counts"]]
    excess = sum(max(0, length - think_threshold) for length in turn_lengths)
    penalty = think_penalty_weight * min(1.0, excess / think_threshold)
    result = grepseek_reward(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        failed_rollout_reward=0.0 if extra_info.get("is_validation") else failed_rollout_reward,
        enable_length_decay=False,
        **{key: value for key, value in kwargs.items() if key != "enable_length_decay"},
    )
    result["metrics/turn_think_max_tokens"] = max(turn_lengths, default=0)
    result["metrics/turn_think_total_tokens"] = sum(turn_lengths)
    result["metrics/turn_think_turn_count"] = len(turn_lengths)
    result["metrics/turn_think_over_threshold_count"] = sum(length > think_threshold for length in turn_lengths)
    result["metrics/turn_think_excess_tokens"] = excess
    result["reward/turn_think_penalty"] = 0.0

    if extra_info.get("is_validation") or extra_info.get("invalid_tool_request"):
        return result
    if _has_no_final_answer(solution_str, bool(extra_info.get("hit_max_context"))):
        result["score"] = failed_rollout_reward
        result["reward/total"] = failed_rollout_reward
        result["reward/failed_rollout"] = 1.0
        result["reward/format_penalty"] = 0.0
        result["metrics/is_no_final_answer"] = float(bool(solution_str.strip()))
        return result
    if not result["reward/format_pass"]:
        return result

    score = max(0.0, result["reward/f1"] - penalty)
    result["score"] = score
    result["reward/total"] = score
    result["reward/length_decay"] = penalty
    result["reward/length_penalty"] = penalty
    result["reward/turn_think_penalty"] = penalty
    return result
