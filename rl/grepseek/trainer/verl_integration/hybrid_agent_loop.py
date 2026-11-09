from __future__ import annotations

import asyncio
import json
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
from grepseek.trainer.verl_integration.invalid_tool_request import (
    InvalidToolRequest,
    classify_invalid_tool_request,
    format_parsed_tool_call,
)
from inference.retrieval_env.schemas import tool_schemas, validate_parameters
from inference.retrieval_env.sessions import RetrievalSession
from verl.experimental.agent_loop.agent_loop import register


@register("grepseek_hybrid_agent")
class HybridAgentLoop(GrepSeekAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_context = ContextVar("grepseek_retrieval_session", default=None)
        self.record_retrieval_rounds = bool(
            getattr(self, "config", {}).get("algorithm", {}).get("answer_body_gate", {}).get("enabled", False)
            or getattr(self, "config", {}).get("algorithm", {}).get("stop_control", {}).get("enabled", False)
        )

    async def _handle_pending_state(self, agent_data, sampling_params):
        if getattr(self, "native_tool_call_format", False):
            active = getattr(agent_data, "_active_tools", self.tools)
            # The generic Pydantic schema omits nested oneOf/minLength fields.
            # The native prompt must expose the same full schema validated by
            # the hybrid tools, including candidate-scope alternatives.
            agent_data._active_tool_schemas = [
                schema for schema in tool_schemas() if schema["function"]["name"] in active
            ]
        dense_tool = self.tools["dense_search"]
        session = RetrievalSession(
            agent_data.request_id, dense_tool.store, dense_tool.ttl_seconds, os.environ.get("GREPSEEK_TMP_ROOT")
        )
        agent_data.retrieval_session = session
        self._session_context.set(session)
        agent_data.extra_fields["retrieval_request_id"] = agent_data.request_id
        agent_data.extra_fields["corpus_version"] = dense_tool.store.version
        if getattr(self, "record_retrieval_rounds", False):
            agent_data.extra_fields["retrieval_rounds"] = []
        state = await super()._handle_pending_state(agent_data, sampling_params)
        if len(agent_data.prompt_ids) > self.prompt_length:
            raise ValueError(f"Hybrid prompt exceeds the unchanged {self.prompt_length}-token prompt budget")
        return state

    async def _handle_processing_tools_state(self, agent_data):
        # Capture only an action that actually reached tool processing. A
        # terminal generation can leave stale tool_calls in the parent, so no
        # pending state may be stored in exported per-trajectory extra_fields.
        # response_ids is the latest sampled generation, excluding observations
        # and environment-prefilled tokens. Its offsets require no re-tokenizing.
        pending = None
        if getattr(self, "record_retrieval_rounds", False) and agent_data.tool_calls:
            end = len(agent_data.response_mask)
            pending = {
                "generated_start": end - len(agent_data.response_ids),
                "generated_end": end,
                "tool_name": agent_data.tool_calls[0].name,
            }
        old_turns = agent_data.user_turns
        state = await super()._handle_processing_tools_state(agent_data)
        # The parent may execute a tool but fail to append an observation when
        # context is full. Such an incomplete round must keep its original credit.
        if pending is not None and agent_data.user_turns == old_turns + 1:
            pending["prefix_end"] = len(agent_data.response_mask)
            pending["round"] = agent_data.user_turns
            agent_data.extra_fields["retrieval_rounds"].append(pending)
        return state

    def _classify_invalid_tool_request(self, response_text, tool_calls, active_tools):
        if not tool_calls:
            return classify_invalid_tool_request(response_text, tool_calls, active_tools)
        if len(tool_calls) != 1:
            return InvalidToolRequest("invalid_arguments", "Exactly one tool call is allowed per turn")
        tool_call = tool_calls[0]
        if tool_call.name not in active_tools:
            return InvalidToolRequest(
                "wrong_tool_name",
                f"Unknown tool {tool_call.name!r}; available tools: {sorted(active_tools)}",
                format_parsed_tool_call(tool_call),
            )
        try:
            arguments = json.loads(tool_call.arguments)
            validate_parameters(arguments, active_tools[tool_call.name].parameter_schema)
        except (ValueError, TypeError) as exc:
            return InvalidToolRequest("invalid_arguments", str(exc), format_parsed_tool_call(tool_call))
        return None

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        response, reward, result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        try:
            payload = json.loads(response.text)
        except (ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        exit_code = result.get("exit_code")
        if exit_code is None:
            exit_code = payload.get("exit_code")
        timed_out = bool(result.get("timed_out") or payload.get("timed_out"))
        status = result.get("status") or payload.get("status")
        if timed_out:
            status = "timeout"
        elif result.get("validation_error"):
            status = "validation_error"
        elif status in (None, "ok"):
            status = (
                "execution_error"
                if exit_code is not None and exit_code not in (0, 1)
                else "ok"
            )
        event = {"tool_name": tool_call.name, **result, "status": status}
        if exit_code is not None:
            event["exit_code"] = exit_code
        if "timed_out" in result or "timed_out" in payload:
            event["timed_out"] = timed_out
        if payload:
            payload["remaining_tool_calls"] = max(0, self.max_user_turns - agent_data.user_turns - 1)
            if tool_call.name == "shell":
                arguments = json.loads(tool_call.arguments)
                payload["scope"] = arguments.get("scope", {"type": "corpus"})
                payload["corpus_version"] = agent_data.retrieval_session.store.version
            response.text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if response.text:
            event["observation_tokens"] = len(self.tokenizer.encode(response.text, add_special_tokens=False))
        agent_data.extra_fields.setdefault("tool_trace", []).append(event)
        counts = agent_data.extra_fields.setdefault("tool_counts", {})
        counts[tool_call.name] = counts.get(tool_call.name, 0) + 1
        return response, reward, result

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        token = self._session_context.set(None)
        try:
            output = await super().run(sampling_params, **kwargs)
            output.extra_fields.setdefault("tool_counts", {})
            output.extra_fields.setdefault("tool_trace", [])
            if getattr(self, "config", {}).get("algorithm", {}).get("rubric", {}).get("enabled", False):
                from grepseek.rubric.core import visible_events

                output.extra_fields["rubric_events"] = visible_events(
                    self.tokenizer, output.response_ids, output.response_mask,
                )
            output_dir = os.environ.get("GREPSEEK_OUTPUT_DIR")
            if output_dir:
                record = {
                    "request_id": output.extra_fields.get("retrieval_request_id"),
                    "qid": (kwargs.get("extra_info") or {}).get("qid"),
                    "is_validation": bool((kwargs.get("extra_info") or {}).get("is_validation")),
                    "tool_counts": output.extra_fields["tool_counts"],
                    "tool_trace": output.extra_fields["tool_trace"],
                    "invalid_tool_request": output.extra_fields["invalid_tool_request"],
                    "invalid_tool_request_reason": output.extra_fields.get("invalid_tool_request_reason", ""),
                    "tool_call_limit_exceeded": output.extra_fields.get("tool_call_limit_exceeded", False),
                    "tool_call_attempt_count": output.extra_fields.get("tool_call_attempt_count", 0),
                    "over_limit_tool_request": output.extra_fields.get("over_limit_tool_request", ""),
                    "over_limit_tool_names": output.extra_fields.get("over_limit_tool_names", []),
                    "generated_token_count": output.extra_fields.get("generated_token_count", 0),
                    "response_token_count": output.extra_fields.get("response_token_count", 0),
                    "forced_final_answer": output.extra_fields.get("forced_final_answer", False),
                    "natural_answer_emitted": output.extra_fields.get("natural_answer_emitted", False),
                    "tool_limit_behavior": output.extra_fields.get("tool_limit_behavior", "force_answer"),
                }
                await asyncio.get_running_loop().run_in_executor(None, self._write_trace, output_dir, record)
            return output
        finally:
            session = self._session_context.get()
            if session is not None:
                session.close()
            self._session_context.reset(token)

    @staticmethod
    def _write_trace(output_dir, record):
        directory = Path(output_dir) / "hybrid_traces"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"worker_{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
