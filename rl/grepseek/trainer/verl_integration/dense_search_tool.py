from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from uuid import uuid4

from inference.retrieval_env.corpus_store import get_corpus_store
from inference.retrieval_env.dense_client import DenseClient, DenseServiceError
from inference.retrieval_env.observations import dense_observation, encode_json
from inference.retrieval_env.schemas import DENSE_TOP_K, validate_parameters
from inference.retrieval_env.sessions import CandidateError
from grepseek.trainer.verl_integration.tool_executors import get_tool_executor


class DenseSearchTool:
    def __init__(self, config: dict[str, Any], tool_schema=None):
        from inference.retrieval_env.schemas import tool_schema as shared_schema
        from verl.tools.schemas import OpenAIFunctionToolSchema

        schema = shared_schema("dense_search")
        self.tool_schema = tool_schema or OpenAIFunctionToolSchema.model_validate(schema)
        self.parameter_schema = schema["function"]["parameters"]
        self.name = "dense_search"
        self.config = config
        corpus_root = os.environ.get("VERL_TOOL_CORPUS_ROOT") or config["corpus_root"]
        data_root = os.environ.get("GREPSEEK_DENSE_DATA_ROOT") or config["data_root"]
        self.store = get_corpus_store(os.path.join(corpus_root, "wiki_corpus.jsonl"), data_root)
        self.client = DenseClient(
            os.environ.get("GREPSEEK_DENSE_URL") or config["url"], float(config.get("timeout_seconds", 180.0))
        )
        self.max_output_tokens = int(config.get("max_output_tokens", 1024))
        self.ttl_seconds = float(config.get("candidate_ttl_seconds", 3600.0))
        self._tokenizer = None
        self._executor = get_tool_executor("dense", int(config.get("executor_workers", 8)))

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def get_openai_tool_schema(self):
        return self.tool_schema

    async def create(self, instance_id=None, **kwargs):
        from verl.tools.schemas import ToolResponse

        return instance_id or uuid4().hex, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        from verl.tools.schemas import ToolResponse

        started = time.monotonic()
        payload = {
            "tool": self.name,
            "requested_k": DENSE_TOP_K,
            "returned_k": 0,
            "displayed_k": 0,
            "documents": [],
            "corpus_version": self.store.version,
            "observation_truncated": False,
        }
        try:
            validate_parameters(parameters, self.parameter_schema)
            agent_data = kwargs.get("agent_data")
            session = getattr(agent_data, "retrieval_session", None)
            if session is None or session.request_id != agent_data.request_id:
                raise CandidateError("invalid_session", "Dense retrieval requires a rollout session")
            if self._tokenizer is None:
                raise RuntimeError("DenseSearchTool requires the actor tokenizer")
            payload = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._retrieve_observation, parameters["query"], session
            )
        except (DenseServiceError, CandidateError) as exc:
            payload.update(status=exc.status, error=str(exc))
        except ValueError as exc:
            payload.update(status="invalid_request", error=str(exc))
        response_text = encode_json(payload)
        metrics = {
            "tool_name": self.name,
            "status": payload["status"],
            "requested_k": DENSE_TOP_K,
            "returned_k": payload["returned_k"],
            "displayed_k": payload["displayed_k"],
            "elapsed_seconds": time.monotonic() - started,
        }
        return ToolResponse(text=response_text), 0.0, metrics

    def _retrieve_observation(self, query, session):
        results = self.client.retrieve(query)
        handle = session.register(results)
        return dense_observation(results, handle, self.store.version, self._tokenizer, self.max_output_tokens)

    async def calc_reward(self, instance_id, **kwargs):
        return 0.0

    async def release(self, instance_id, **kwargs):
        return None
