from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from grepseek.trainer.verl_integration.search_corpus_tool import SearchCorpusTool, ToolError
from inference.retrieval_env.candidate_executor import execute_candidate_pipeline
from inference.retrieval_env.schemas import validate_parameters
from inference.retrieval_env.sessions import CandidateError


class HybridShellTool(SearchCorpusTool):
    def __init__(self, config: dict[str, Any], tool_schema=None):
        from inference.retrieval_env.schemas import tool_schema as shared_schema
        from verl.tools.schemas import OpenAIFunctionToolSchema

        schema = shared_schema("shell")
        super().__init__(config, tool_schema or OpenAIFunctionToolSchema.model_validate(schema))
        self.parameter_schema = schema["function"]["parameters"]

    async def execute(self, instance_id, parameters, **kwargs):
        from verl.tools.schemas import ToolResponse

        started = time.monotonic()
        scope = parameters.get("scope", {"type": "corpus"}) if isinstance(parameters, dict) else {"type": "corpus"}
        payload = {
            "command": parameters.get("command", "") if isinstance(parameters, dict) else "",
            "stdout": "",
            "stderr": "",
            "exit_code": -2,
            "timed_out": False,
            "truncated": False,
            "scope": scope,
        }
        try:
            validate_parameters(parameters, self.parameter_schema)
            if scope["type"] == "corpus":
                response, reward, metrics = await super().execute(
                    instance_id, {"command": parameters["command"]}, **kwargs
                )
                metrics.update(tool_name="shell", scope="corpus", elapsed_seconds=time.monotonic() - started)
                return response, reward, metrics
            agent_data = kwargs.get("agent_data")
            session = getattr(agent_data, "retrieval_session", None)
            if session is None:
                raise CandidateError("invalid_candidate_set", "Candidate scope requires a rollout session")
            candidates = session.resolve(scope["candidate_set_id"], agent_data.request_id)
            if self._tokenizer is None:
                raise RuntimeError("HybridShellTool requires the actor tokenizer")
            payload = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                lambda: execute_candidate_pipeline(
                    parameters["command"],
                    candidates.path,
                    self.timeout_seconds,
                    self.max_output_tokens,
                    self._tokenizer,
                ),
            )
            payload.update(
                scope=scope,
                corpus_version=session.store.version,
                candidate_count=len(candidates.passage_ids),
                source_passage_ids=candidates.passage_ids,
            )
        except CandidateError as exc:
            payload.update(status=exc.status, stderr=str(exc))
        except (ValueError, ToolError) as exc:
            payload.update(status="invalid_scope_command", stderr=str(exc))
        metrics = {
            "tool_name": "shell",
            "scope": scope.get("type", "invalid") if isinstance(scope, dict) else "invalid",
            "status": payload.get("status", "execution_error"),
            "exit_code": payload["exit_code"],
            "timed_out": payload["timed_out"],
            "truncated": payload["truncated"],
            "elapsed_seconds": time.monotonic() - started,
        }
        return ToolResponse(text=json.dumps(payload, ensure_ascii=False)), 0.0, metrics
