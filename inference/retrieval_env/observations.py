from __future__ import annotations

import json
from typing import Any

from inference.retrieval_env.schemas import DENSE_TOP_K


def encode_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def dense_observation(
    results: list[dict[str, Any]], handle: str, corpus_version: str, tokenizer: Any, max_tokens: int
) -> dict[str, Any]:
    documents = []
    content_ids = []
    for result in results:
        document = result["document"]
        identifier = str(document["id"])
        documents.append(
            {
                "passage_id": identifier,
                "title": document.get(
                    "title", document["contents"].partition("\n")[0] if "\n" in document["contents"] else ""
                ),
                "contents": document["contents"],
                "rank": result["rank"],
                "score": result["score"],
                "source": f"{corpus_version}:{identifier}",
                "content_truncated": False,
            }
        )
        content_ids.append(tokenizer.encode(document["contents"], add_special_tokens=False))
    payload = {
        "tool": "dense_search",
        "status": "ok" if documents else "no_hits",
        "corpus_version": corpus_version,
        "candidate_set_id": handle,
        "requested_k": DENSE_TOP_K,
        "returned_k": len(results),
        "displayed_k": len(documents),
        "documents": documents,
        "observation_truncated": False,
    }
    if len(tokenizer.encode(encode_json(payload), add_special_tokens=False)) <= max_tokens:
        return payload

    def clipped(limit: int) -> dict[str, Any]:
        visible = []
        for document, token_ids in zip(documents, content_ids, strict=True):
            entry = dict(document)
            if len(token_ids) > limit:
                entry["contents"] = tokenizer.decode(token_ids[:limit], skip_special_tokens=False)
                entry["content_truncated"] = True
            visible.append(entry)
        return {
            **payload,
            "documents": visible,
            "displayed_k": sum(bool(entry["contents"]) for entry in visible),
            "observation_truncated": True,
        }

    best = clipped(0)
    if len(tokenizer.encode(encode_json(best), add_special_tokens=False)) > max_tokens:
        raise ValueError("Dense observation metadata exceeds its token budget")
    lower = 0
    upper = max_tokens
    while lower <= upper:
        limit = (lower + upper) // 2
        candidate = clipped(limit)
        if len(tokenizer.encode(encode_json(candidate), add_special_tokens=False)) <= max_tokens:
            best = candidate
            lower = limit + 1
        else:
            upper = limit - 1
    return best
