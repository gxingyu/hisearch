from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from inference.retrieval_env.schemas import DENSE_TOP_K


class DenseServiceError(Exception):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status


class DenseClient:
    def __init__(self, url: str, timeout_seconds: float = 180.0):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Dense retrieval URL must be an HTTP(S) endpoint")
        self.url = url
        self.health_url = urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
        self.timeout_seconds = timeout_seconds

    def _request(self, url: str, payload: dict | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=self.timeout_seconds) as response:
                contents = response.read(2 * 1024 * 1024 + 1)
        except HTTPError as exc:
            status = "service_unavailable" if exc.code in (502, 503, 504) else "service_error"
            raise DenseServiceError(status, f"Dense service returned HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise DenseServiceError("timeout", "Dense service request timed out") from exc
        except URLError as exc:
            status = "timeout" if isinstance(exc.reason, TimeoutError) else "service_unavailable"
            raise DenseServiceError(status, f"Dense service request failed: {exc.reason}") from exc
        except OSError as exc:
            raise DenseServiceError("service_unavailable", f"Dense connection failed: {exc}") from exc
        if len(contents) > 2 * 1024 * 1024:
            raise DenseServiceError("protocol_error", "Dense response exceeds 2 MiB")
        try:
            result = json.loads(contents)
        except (ValueError, UnicodeError) as exc:
            raise DenseServiceError("protocol_error", "Dense service returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise DenseServiceError("protocol_error", "Dense service response must be an object")
        return result

    def health(self) -> dict[str, Any]:
        health = self._request(self.health_url)
        if health.get("status") != "ok" or not health.get("corpus_sha256"):
            raise DenseServiceError("protocol_error", "Endpoint is not a ready Search-R1 retrieval service")
        return health

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        response = self._request(self.url, {"queries": [query], "topk": DENSE_TOP_K, "return_scores": True})
        batches = response.get("result")
        if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
            raise DenseServiceError("protocol_error", "Dense response must contain one result batch")
        if len(batches[0]) > DENSE_TOP_K:
            raise DenseServiceError("protocol_error", "Dense service returned more than 3 passages")
        documents = []
        for rank, item in enumerate(batches[0], 1):
            document = item.get("document") if isinstance(item, dict) else None
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(document, dict) or "id" not in document or not isinstance(document.get("contents"), str):
                raise DenseServiceError("protocol_error", "Dense result is missing passage ID or contents")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise DenseServiceError("protocol_error", "Dense result has an invalid score")
            documents.append({"document": document, "score": float(score), "rank": rank})
        return documents
