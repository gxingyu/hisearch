# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
"""
Custom VERL agent loop for grepseek GRPO training.

Subclasses ToolAgentLoop with three modifications:
1. _handle_pending_state: passes tools=[] to apply_chat_template so the tool spec
   is NOT injected via Qwen3's native tool calling (it is already embedded in the
   system prompt text, identical to the inference pipeline).
2. At the tool-call budget, prefills the answer block and gives the final
   generation a separate token budget.
3. run: records len(output.response_ids) as "response_token_count" in extra_fields
   so the reward function can compute the length-decay f(L).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vllm.sampling_params import StructuredOutputsParams

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentLoopOutput,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.rollout_trace import rollout_trace_op

from grepseek.trainer.verl_integration.invalid_tool_request import (
    InvalidToolRequest,
    append_terminal_invalid_tool_response,
    classify_invalid_tool_request,
    extract_attempted_tool_request,
    looks_like_attempted_tool_call,
)

logger = logging.getLogger(__name__)

_FINAL_ANSWER_INSTRUCTION = (
    "Retrieval budget exhausted. Do not emit another tool call. "
    "Use the evidence already retrieved and finish now with the single best "
    "answer in the format required by the system prompt."
)
_FINAL_ANSWER_REGEX = r"[^<]*</think>[ \t\r\n]*<answer>[ \t\r\n]*[^< \t\r\n][^<]*</answer>"
_FINAL_ANSWER_PREFIX = "</think>\n<answer>\n"
_FINAL_ANSWER_CONTENT_REGEX = r"[^< \t\r\n][^<]{0,255}</answer>"


def _count_turn_think_tokens(token_ids: list[int], tokenizer: Any) -> int:
    def marker_index(marker: str) -> int | None:
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        width = len(marker_ids)
        for position in range(len(token_ids) - width + 1):
            if token_ids[position : position + width] == marker_ids:
                return position
        return None

    end = marker_index("</think>")
    if end is None:
        actions = [
            position
            for marker in ("<tool_call>", "<answer>")
            if (position := marker_index(marker)) is not None
        ]
        end = min(actions, default=len(token_ids))
    opening = marker_index("<think>")
    start = 0
    if opening is not None and opening < end:
        start = opening + len(tokenizer.encode("<think>", add_special_tokens=False))
    return max(0, end - start)


def _add_final_answer_instruction(response_text: str | None) -> str:
    if not response_text:
        return _FINAL_ANSWER_INSTRUCTION
    try:
        payload = json.loads(response_text)
    except (TypeError, json.JSONDecodeError):
        return f"{response_text}\n{_FINAL_ANSWER_INSTRUCTION}"
    if not isinstance(payload, dict):
        return f"{response_text}\n{_FINAL_ANSWER_INSTRUCTION}"
    payload["final_answer_required"] = _FINAL_ANSWER_INSTRUCTION
    return json.dumps(payload, ensure_ascii=False)


@register("grepseek_agent")
class GrepSeekAgentLoop(ToolAgentLoop):
    """Agent loop for grepseek that keeps the tool spec in the system prompt.

    The system prompt already contains the shell tool spec (built identically
    to the inference pipeline via render_system_prompt). Passing tools=[] prevents
    the Qwen3 chat template from injecting a duplicate tool spec block.
    """

    def __init__(self, *args, **kwargs):
        self.native_tool_call_format = bool(kwargs.pop("native_tool_call_format", False))
        self.tool_limit_behavior = kwargs.pop("tool_limit_behavior", "force_answer")
        if self.tool_limit_behavior not in ("force_answer", "truncate_invalid"):
            raise ValueError(f"Unknown tool_limit_behavior: {self.tool_limit_behavior!r}")
        self.force_final_answer_at_tool_limit = bool(
            kwargs.pop("force_final_answer_at_tool_limit", True)
        )
        self.prefill_final_answer_at_tool_limit = bool(
            kwargs.pop("prefill_final_answer_at_tool_limit", True)
        )
        if self.tool_limit_behavior == "truncate_invalid":
            self.force_final_answer_at_tool_limit = False
            self.prefill_final_answer_at_tool_limit = False
        self.final_answer_max_tokens = int(kwargs.pop("final_answer_max_tokens", 256))
        if self.final_answer_max_tokens <= 0:
            raise ValueError("final_answer_max_tokens must be positive")
        super().__init__(*args, **kwargs)
        self.final_answer_prefix_ids = self.tokenizer.encode(
            _FINAL_ANSWER_PREFIX, add_special_tokens=False
        )
        # SearchCorpusTool needs the trained model's tokenizer to apply the
        # configured token-based stdout limit. The agent loop already loaded it from
        # actor_rollout_ref.model.path during super().__init__, so just hand it
        # to each tool that asks for it. Zero config duplication.
        for tool in self.tools.values():
            if hasattr(tool, "set_tokenizer"):
                tool.set_tokenizer(self.tokenizer)

    async def process_vision_info(self, messages: list[dict]) -> dict:
        # Text-only task: skip image/video processing regardless of processor.
        return {}

    async def _handle_pending_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        """Inject native schemas only for an explicitly enabled native-format run."""
        tools = []
        if getattr(self, "native_tool_call_format", False):
            tools = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=tools,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    def _prepare_final_answer_generation(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
    ) -> dict[str, Any] | None:
        prefix_ids = self.final_answer_prefix_ids if self.prefill_final_answer_at_tool_limit else []
        response_budget = self.response_length - len(agent_data.response_mask) - len(prefix_ids)
        model_length = self.rollout_config.max_model_len or self.prompt_length + self.response_length
        context_budget = model_length - len(agent_data.prompt_ids) - len(prefix_ids)
        final_budget = min(self.final_answer_max_tokens, response_budget, context_budget)
        if final_budget <= 0:
            agent_data.extra_fields["final_answer_budget_exhausted"] = True
            return None

        params = dict(sampling_params)
        requested_budget = params.pop("max_new_tokens", params.get("max_tokens", final_budget))
        params["max_tokens"] = min(final_budget, requested_budget)
        stops = params.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        if "</answer>" not in stops:
            stops.append("</answer>")
        params["stop"] = stops
        params["include_stop_str_in_output"] = True
        regex = _FINAL_ANSWER_CONTENT_REGEX if prefix_ids else _FINAL_ANSWER_REGEX
        params["structured_outputs"] = StructuredOutputsParams(regex=regex)

        agent_data.prompt_ids += prefix_ids
        agent_data.response_mask += [0] * len(prefix_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(prefix_ids)
        agent_data.extra_fields["final_answer_prefill_tokens"] = len(prefix_ids)
        agent_data.extra_fields["final_answer_max_tokens"] = params["max_tokens"]
        return params

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        force_final_answer = bool(
            getattr(self, "tool_limit_behavior", "force_answer") != "truncate_invalid"
            and agent_data.extra_fields.get("force_final_answer_next_turn", False)
        )
        if force_final_answer:
            agent_data.extra_fields["forced_final_answer"] = True
            sampling_params = self._prepare_final_answer_generation(agent_data, sampling_params)
            if sampling_params is None:
                return AgentState.TERMINATED

        response_budget = self.response_length - len(agent_data.response_mask)
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        generated_ids = agent_data.response_ids[:response_budget]
        turn_think_tokens = (
            0
            if force_final_answer and self.prefill_final_answer_at_tool_limit
            else _count_turn_think_tokens(generated_ids, self.tokenizer)
        )
        agent_data.extra_fields.setdefault("turn_think_token_counts", []).append(turn_think_tokens)
        if force_final_answer:
            agent_data.extra_fields["final_answer_hit_token_limit"] = (
                len(agent_data.response_ids) >= agent_data.extra_fields["final_answer_max_tokens"]
            )

        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
        )
        tool_calls = agent_data.tool_calls if state == AgentState.PROCESSING_TOOLS else []
        if not tool_calls and "<tool_call" in response_text:
            active_tools = getattr(agent_data, "_active_tools", self.tools)
            tools = [tool.tool_schema for tool in active_tools.values()]
            _, tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        # The parent checks both turn limits before parsing the last generation.
        # Inspect that generation even when it already returned TERMINATED.
        if self._tool_budget_exhausted(agent_data) and (
            tool_calls or looks_like_attempted_tool_call(response_text)
        ):
            return self._terminate_over_budget_tool_request(agent_data, response_text, tool_calls)
        invalid = self._classify_invalid_tool_request(
            response_text,
            tool_calls,
            getattr(agent_data, "_active_tools", self.tools),
        )
        if invalid is None:
            if not tool_calls:
                agent_data.extra_fields["natural_answer_emitted"] = bool(
                    not force_final_answer and "<answer>" in response_text and "</answer>" in response_text
                )
            return state

        return await self._terminate_invalid_tool_request(agent_data, invalid)

    def _tool_budget_exhausted(self, agent_data: AgentData) -> bool:
        return bool(
            getattr(self, "tool_limit_behavior", "force_answer") == "truncate_invalid"
            and self.max_user_turns
            and agent_data.user_turns >= self.max_user_turns
        )

    def _terminate_over_budget_tool_request(
        self, agent_data: AgentData, response_text: str, tool_calls: list[FunctionCall]
    ) -> AgentState:
        # Keep sampled tokens and logprobs intact. A rejected sixth request is
        # not an executed tool call and must not append a synthetic observation.
        agent_data.extra_fields.update(
            invalid_tool_request=True,
            invalid_tool_call_count=int(agent_data.extra_fields.get("invalid_tool_call_count", 0)) + 1,
            invalid_tool_request_reason="tool_call_limit_exceeded",
            invalid_tool_request_detail=f"Tool-call budget of {self.max_user_turns} exhausted",
            tool_call_limit_exceeded=True,
            over_limit_tool_request=extract_attempted_tool_request(response_text),
            over_limit_tool_names=[call.name for call in tool_calls],
            tool_call_attempt_count=int(agent_data.extra_fields.get("tool_call_attempt_count", 0)) + 1,
        )
        agent_data.tool_calls = []
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        # Defense at dispatch as well as at generation; never execute call six.
        if self._tool_budget_exhausted(agent_data) and agent_data.tool_calls:
            response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            return self._terminate_over_budget_tool_request(agent_data, response_text, agent_data.tool_calls)
        return await super()._handle_processing_tools_state(agent_data)

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        agent_data.extra_fields["tool_call_attempt_count"] = (
            int(agent_data.extra_fields.get("tool_call_attempt_count", 0)) + 1
        )
        response, reward, result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        reaches_tool_limit = bool(
            self.force_final_answer_at_tool_limit
            and self.max_user_turns
            and agent_data.user_turns + 1 >= self.max_user_turns
        )
        if reaches_tool_limit:
            response.text = _add_final_answer_instruction(response.text)
            agent_data.extra_fields["force_final_answer_next_turn"] = True
        return response, reward, result

    def _classify_invalid_tool_request(
        self,
        response_text: str,
        tool_calls: list[FunctionCall],
        active_tools: dict[str, Any],
    ) -> InvalidToolRequest | None:
        return classify_invalid_tool_request(response_text, tool_calls, active_tools)

    async def _terminate_invalid_tool_request(
        self,
        agent_data: AgentData,
        invalid: InvalidToolRequest,
    ) -> AgentState:
        agent_data.extra_fields["tool_call_attempt_count"] = (
            int(agent_data.extra_fields.get("tool_call_attempt_count", 0)) + 1
        )
        await append_terminal_invalid_tool_response(
            agent_data,
            invalid,
            apply_chat_template=self.apply_chat_template,
            response_length=self.response_length,
        )
        return AgentState.TERMINATED

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> AgentLoopOutput:
        """Run the agent loop and record response token stats in extra_fields."""
        output: AgentLoopOutput = await super().run(sampling_params, **kwargs)
        n_tokens = len(output.response_ids)
        output.extra_fields["response_token_count"] = n_tokens
        # The mask counts only sampled model tokens, excluding observations and
        # injected final-answer prefixes. This is reward metadata only.
        output.extra_fields["generated_token_count"] = sum(output.response_mask)
        output.extra_fields.setdefault("tool_call_attempt_count", 0)
        output.extra_fields.setdefault("turn_think_token_counts", [])
        # Flag whether the trajectory was cut off at the response_length budget.
        # The reward function uses this to distinguish a natural termination from
        # a hard truncation (model was still mid-search when context ran out).
        output.extra_fields["hit_max_context"] = n_tokens >= self.response_length
        output.extra_fields.setdefault("invalid_tool_request", False)
        output.extra_fields.setdefault("invalid_tool_call_count", 0)
        output.extra_fields.setdefault("invalid_tool_request_reason", "")
        output.extra_fields.setdefault("invalid_tool_request_detail", "")
        output.extra_fields.setdefault("tool_call_limit_exceeded", False)
        output.extra_fields.setdefault("over_limit_tool_request", "")
        output.extra_fields.setdefault("over_limit_tool_names", [])
        output.extra_fields.setdefault("natural_answer_emitted", False)
        output.extra_fields["tool_call_limit"] = self.max_user_turns
        output.extra_fields["tool_limit_behavior"] = getattr(self, "tool_limit_behavior", "force_answer")
        output.extra_fields.setdefault("forced_final_answer", False)
        output.extra_fields.setdefault("force_final_answer_next_turn", False)
        output.extra_fields.setdefault("final_answer_budget_exhausted", False)
        output.extra_fields.setdefault("final_answer_hit_token_limit", False)
        output.extra_fields.setdefault("final_answer_prefill_tokens", 0)
        output.extra_fields.setdefault("final_answer_max_tokens", 0)
        return output
