from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from grepseek.trainer.verl_integration.agent_loop import GrepSeekAgentLoop
from grepseek.trainer.verl_integration.dense_search_tool import DenseSearchTool
from grepseek.trainer.verl_integration.hybrid_agent_loop import HybridAgentLoop
from grepseek.trainer.verl_integration.hybrid_shell_tool import HybridShellTool
from grepseek.trainer.verl_integration.search_corpus_tool import SearchCorpusTool

from inference.retrieval_env.candidate_executor import candidate_pipeline, execute_candidate_pipeline
from inference.retrieval_env.corpus_store import CorpusStore
from inference.retrieval_env.dense_client import DenseClient, DenseServiceError
from inference.retrieval_env.observations import dense_observation, encode_json
from inference.retrieval_env.preflight import check_prompts
from inference.retrieval_env.schemas import tool_schema, validate_parameters
from inference.retrieval_env.sessions import CandidateError, RetrievalSession
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.tool_parser import FunctionCall


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


class RetrievalHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"result": [self.server.results]}).encode())

    def log_message(self, format, *args):
        return None


class HybridToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.documents = [
            {"id": "0", "contents": '"Marie Curie"\nMarie Curie married Pierre Curie. Rügen is an island.'},
            {"id": "1", "contents": '"Other article"\nAn unrelated passage.'},
            {"id": "2", "contents": '"Pierre Curie"\nPierre Curie was Marie Curie\'s spouse.'},
        ]
        self.records = [(json.dumps(document, ensure_ascii=True) + "\n").encode() for document in self.documents]
        corpus = self.root / "wiki_corpus.jsonl"
        corpus.write_bytes(b"".join(self.records))
        offsets = [0]
        for record in self.records:
            offsets.append(offsets[-1] + len(record))
        np.save(self.root / "corpus_offsets.npy", np.array(offsets, dtype=np.uint64))
        manifest = {
            "path": str(corpus),
            "size": corpus.stat().st_size,
            "mtime_ns": corpus.stat().st_mtime_ns,
            "count": 3,
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "ids": "sequential strings, checked on every row",
        }
        (self.root / "corpus_manifest.json").write_text(json.dumps(manifest))
        self.store = CorpusStore(corpus, self.root)
        self.results = [
            {"document": self.documents[index], "score": 0.9 - rank / 10, "rank": rank + 1}
            for rank, index in enumerate((2, 0, 1))
        ]
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RetrievalHandler)
        self.server.requests = []
        self.server.results = [{"document": result["document"], "score": result["score"]} for result in self.results]
        self.server.status = 200
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/retrieve"
        self.environment = patch.dict(
            os.environ,
            {
                "VERL_TOOL_CORPUS_ROOT": str(self.root),
                "GREPSEEK_DENSE_DATA_ROOT": str(self.root),
                "GREPSEEK_DENSE_URL": self.url,
                "GREPSEEK_SEARCH_ENGINE": "none",
                "GREPSEEK_OUTPUT_DIR": "",
            },
        )
        self.environment.start()
        self.tokenizer = CharacterTokenizer()
        config = {
            "corpus_root": str(self.root),
            "data_root": str(self.root),
            "url": self.url,
            "max_output_tokens": 2048,
        }
        self.dense = DenseSearchTool(config)
        self.shell = HybridShellTool(config)
        self.dense.set_tokenizer(self.tokenizer)
        self.shell.set_tokenizer(self.tokenizer)
        self.session = RetrievalSession("request-a", self.store)
        self.data = AgentData([], None, None, {}, "request-a", {})
        self.data.retrieval_session = self.session

    def tearDown(self):
        self.session.close()
        self.environment.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    async def test_dense_then_candidate_shell_survives_tool_release(self):
        instance, _ = await self.dense.create()
        response, reward, _ = await self.dense.execute(
            instance, {"query": "Marie Curie's spouse?"}, agent_data=self.data
        )
        await self.dense.release(instance)
        payload = json.loads(response.text)
        self.assertEqual(
            self.server.requests, [{"queries": ["Marie Curie's spouse?"], "topk": 3, "return_scores": True}]
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(payload["returned_k"], 3)
        candidates = self.session.resolve(payload["candidate_set_id"], "request-a")
        self.assertEqual(candidates.path.read_bytes(), self.records[2] + self.records[0] + self.records[1])
        response, reward, metrics = await self.shell.execute(
            "shell-instance",
            {
                "command": "cat corpus.jsonl | head -n 2",
                "scope": {"type": "candidates", "candidate_set_id": payload["candidate_set_id"]},
            },
            agent_data=self.data,
        )
        shell_payload = json.loads(response.text)
        self.assertEqual(shell_payload["stdout"].encode(), self.records[2] + self.records[0])
        self.assertEqual(metrics["scope"], "candidates")
        self.assertEqual(reward, 0.0)

    async def test_fixed_topk_rejects_model_override_before_http(self):
        for key in ("k", "topk"):
            response, _, _ = await self.dense.execute("instance", {"query": "Curie", key: 10}, agent_data=self.data)
            self.assertEqual(json.loads(response.text)["status"], "invalid_request")
        self.assertEqual(self.server.requests, [])

    async def test_dense_runs_while_shell_and_default_executors_are_busy(self):
        loop = asyncio.get_running_loop()
        blocked = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as default_executor, ThreadPoolExecutor(max_workers=1) as shell_executor:
            loop.set_default_executor(default_executor)
            self.shell._executor = shell_executor
            pending = [
                loop.run_in_executor(default_executor, blocked.wait),
                loop.run_in_executor(shell_executor, blocked.wait),
            ]
            try:
                response, _, _ = await asyncio.wait_for(
                    self.dense.execute("instance", {"query": "Curie"}, agent_data=self.data), timeout=2
                )
                self.assertEqual(json.loads(response.text)["status"], "ok")
            finally:
                blocked.set()
                await asyncio.gather(*pending)

    async def test_service_failure_is_an_observation(self):
        self.server.status = 503
        response, reward, _ = await self.dense.execute("instance", {"query": "Curie"}, agent_data=self.data)
        self.assertEqual(json.loads(response.text)["status"], "service_unavailable")
        self.assertEqual(reward, 0.0)
        with patch.object(self.dense.client, "retrieve", side_effect=DenseServiceError("timeout", "test timeout")):
            response, _, _ = await self.dense.execute("instance", {"query": "Curie"}, agent_data=self.data)
        self.assertEqual(json.loads(response.text)["status"], "timeout")

    async def test_empty_retrieval_has_explicit_no_hits_status(self):
        self.server.results = []
        response, _, _ = await self.dense.execute("instance", {"query": "missing"}, agent_data=self.data)
        payload = json.loads(response.text)
        self.assertEqual(payload["status"], "no_hits")
        self.assertEqual(payload["returned_k"], 0)

    def test_client_rejects_excess_results_and_negative_passage_ids(self):
        self.server.results *= 2
        with self.assertRaisesRegex(DenseServiceError, "more than 3"):
            DenseClient(self.url).retrieve("Curie")
        invalid = [{"document": {"id": "-1", "contents": "wrong"}, "rank": 1, "score": 0.5}]
        with self.assertRaisesRegex(CandidateError, "outside the corpus"):
            self.session.register(invalid)

    def test_handles_are_isolated_expire_and_cleanup(self):
        handle = self.session.register(self.results)
        candidates = self.session.resolve(handle, "request-a")
        with self.assertRaises(CandidateError):
            self.session.resolve(handle, "request-b")
        self.session.ttl_seconds = -1
        with self.assertRaisesRegex(CandidateError, "expired"):
            self.session.resolve(handle, "request-a")
        self.assertFalse(candidates.path.exists())
        directory = candidates.path.parent.parent
        self.session.close()
        self.assertFalse(directory.exists())

    def test_candidate_scope_rejects_file_escape_and_write_options(self):
        handle = self.session.register(self.results)
        path = self.session.resolve(handle, "request-a").path
        for command in (
            "cat /etc/passwd",
            "cat ../corpus.jsonl",
            "rg Curie /etc/passwd",
            "rg --pre cat Curie corpus.jsonl",
            "sort -o corpus.jsonl corpus.jsonl",
            "uniq corpus.jsonl corpus.jsonl",
            "awk '{print}' corpus.jsonl",
            "rg Curie",
            "cat corpus.jsonl | cat /etc/passwd",
        ):
            with self.subTest(command=command), self.assertRaises(CandidateError):
                candidate_pipeline(command, path)

    def test_candidate_search_preserves_unicode_and_count_scope(self):
        handle = self.session.register(self.results)
        path = self.session.resolve(handle, "request-a").path
        result = execute_candidate_pipeline("rg -F 'Rügen' corpus.jsonl | head -n 3", path, 5, 2048, self.tokenizer)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(result["status"], "no_hits")
        result = execute_candidate_pipeline("rg -F 'Curie' corpus.jsonl | wc -l", path, 5, 2048, self.tokenizer)
        self.assertEqual(result["stdout"], "2\n")
        self.assertEqual(result["status"], "ok")

    async def test_full_corpus_shell_reuses_existing_executor(self):
        from verl.tools.schemas import ToolResponse

        expected = ToolResponse(text='{"stdout":"legacy","exit_code":0}')
        with patch.object(SearchCorpusTool, "execute", return_value=(expected, 0.0, {"exit_code": 0})) as execute:
            response, _, _ = await self.shell.execute("instance", {"command": "cat corpus.jsonl"}, agent_data=self.data)
        self.assertIs(response, expected)
        self.assertEqual(execute.call_args.args, ("instance", {"command": "cat corpus.jsonl"}))

    async def test_shell_validation_failure_has_error_metrics(self):
        response, _, metrics = await self.shell.execute(
            "instance", {"command": "rg -f corpus.jsonl"}, agent_data=self.data
        )
        self.assertEqual(json.loads(response.text)["exit_code"], -2)
        self.assertEqual(metrics["exit_code"], -2)
        self.assertEqual(metrics["status"], "validation_error")

    async def test_trace_uses_payload_errors_when_metrics_omit_exit_code(self):
        from verl.tools.schemas import ToolResponse

        agent = object.__new__(HybridAgentLoop)
        agent.tokenizer = self.tokenizer
        agent.max_user_turns = 5
        call = FunctionCall(name="shell", arguments='{"command":"cat corpus.jsonl"}')
        cases = (
            ({"exit_code": -2}, {"validation_error": "bad command"}, "validation_error"),
            ({"exit_code": -2}, {}, "execution_error"),
            ({"exit_code": 2}, {"status": "ok"}, "execution_error"),
            ({"exit_code": -1, "timed_out": True}, {}, "timeout"),
            ({"exit_code": 1, "status": "no_hits"}, {}, "no_hits"),
            ({"exit_code": 0}, {}, "ok"),
        )
        for payload, metrics, expected in cases:
            with self.subTest(expected=expected):
                response = ToolResponse(text=json.dumps(payload))
                with patch.object(GrepSeekAgentLoop, "_call_tool", new=AsyncMock(return_value=(response, 0.0, metrics))):
                    await agent._call_tool(call, {}, self.data)
                event = self.data.extra_fields["tool_trace"][-1]
                self.assertEqual(event["status"], expected)
                self.assertEqual(event["exit_code"], payload["exit_code"])
                self.assertEqual(json.loads(response.text)["remaining_tool_calls"], 4)

    def test_dense_truncation_keeps_valid_json_and_all_candidate_ids(self):
        results = [
            {**result, "document": {**result["document"], "contents": "long passage " * 1000}}
            for result in self.results
        ]
        payload = dense_observation(results, "handle", self.store.version, self.tokenizer, 1024)
        self.assertLessEqual(len(self.tokenizer.encode(encode_json(payload))), 1024)
        self.assertEqual(len(payload["documents"]), 3)
        self.assertEqual(payload["returned_k"], 3)
        self.assertTrue(payload["observation_truncated"])
        self.assertTrue(all(document["content_truncated"] for document in payload["documents"]))

    def test_registered_tool_validation_accepts_dense_and_rejects_bad_scope(self):
        agent = object.__new__(HybridAgentLoop)
        tools = {"shell": self.shell, "dense_search": self.dense}
        dense_call = FunctionCall(name="dense_search", arguments='{"query":"Curie"}')
        self.assertIsNone(agent._classify_invalid_tool_request("", [dense_call], tools))
        self.assertIsNotNone(agent._classify_invalid_tool_request("", [dense_call], {"shell": self.shell}))
        with self.assertRaises(ValueError):
            validate_parameters(
                {"command": "cat corpus.jsonl", "scope": {"type": "candidates"}},
                tool_schema("shell")["function"]["parameters"],
            )

    def test_prompt_check_reads_every_item_and_handles_transformers_mappings(self):
        questions = self.root / "qa.jsonl"
        records = [
            {"id": "short-one", "question": "Who?", "golden_answers": ["name"]},
            {"id": "short-two", "question": "When?", "golden_answers": ["date"]},
            {"id": "long-third", "question": "question " * 100, "golden_answers": ["answer"]},
        ]
        questions.write_text("\n".join(json.dumps(record) for record in records))

        class MappingTokenizer:
            def apply_chat_template(self, conversations, **kwargs):
                return SimpleNamespace(input_ids=[[0] * len(messages[-1]["content"]) for messages in conversations])

        with self.assertRaisesRegex(ValueError, "long-third"):
            check_prompts(MappingTokenizer(), "System", [str(questions)], 64)

    async def test_tool_observation_mask_and_combined_call_budget(self):
        agent = object.__new__(HybridAgentLoop)
        agent.tools = {"shell": self.shell, "dense_search": self.dense}
        agent.tokenizer = self.tokenizer
        agent.max_parallel_calls = 1
        agent.max_user_turns = 5
        agent.max_tool_response_length = 100000
        agent.tool_response_truncate_side = "right"
        agent.tool_parser_name = "hermes"
        agent.force_final_answer_at_tool_limit = True
        self.data.tool_calls = [FunctionCall(name="dense_search", arguments='{"query":"Curie"}')]

        async def template(messages, **kwargs):
            return self.tokenizer.encode(messages[0]["content"])

        agent.apply_chat_template = template
        agent.response_length = 20000
        state = await agent._handle_processing_tools_state(self.data)
        self.assertEqual(state, AgentState.GENERATING)
        self.assertTrue(self.data.response_mask)
        self.assertTrue(all(mask == 0 for mask in self.data.response_mask))
        self.assertEqual(json.loads(self.data.messages[-1]["content"])["remaining_tool_calls"], 4)
        self.assertEqual(self.data.extra_fields["tool_counts"], {"dense_search": 1})
        self.data.user_turns = 4
        response, _, _ = await agent._call_tool(self.data.tool_calls[0], {}, self.data)
        self.assertEqual(json.loads(response.text)["remaining_tool_calls"], 0)
        self.assertTrue(self.data.extra_fields["force_final_answer_next_turn"])

    async def test_session_cleanup_on_success_and_failure(self):
        agent = object.__new__(HybridAgentLoop)
        agent._session_context = ContextVar("test_session", default=None)
        created = []

        async def parent_run(agent_loop, sampling_params, **kwargs):
            session = RetrievalSession(kwargs["request_id"], self.store)
            session.register(self.results)
            agent_loop._session_context.set(session)
            created.append(session)
            await asyncio.sleep(0)
            if kwargs.get("fail"):
                raise RuntimeError("generation failed")
            self.assertEqual(agent_loop._session_context.get().request_id, kwargs["request_id"])
            return SimpleNamespace(extra_fields={"invalid_tool_request": False})

        with patch.object(GrepSeekAgentLoop, "run", parent_run):
            await asyncio.gather(agent.run({}, request_id="one"), agent.run({}, request_id="two"))
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                await agent.run({}, request_id="three", fail=True)
        self.assertTrue(all(session.closed for session in created))
        self.assertIsNone(agent._session_context.get())


if __name__ == "__main__":
    unittest.main()
