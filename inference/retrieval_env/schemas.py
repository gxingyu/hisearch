from __future__ import annotations

from copy import deepcopy
from typing import Any

DENSE_TOP_K = 3

_SCHEMAS = {
    "shell": {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Search corpus.jsonl with a shell pipeline. Optional scope selects the full corpus "
                "or a dense candidate set; omitted scope means corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "scope": {
                        "type": "object",
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"type": {"type": "string", "enum": ["corpus"]}},
                                "required": ["type"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["candidates"]},
                                    "candidate_set_id": {"type": "string", "minLength": 1},
                                },
                                "required": ["type", "candidate_set_id"],
                                "additionalProperties": False,
                            },
                        ],
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    "dense_search": {
        "type": "function",
        "function": {
            "name": "dense_search",
            "description": (
                "Search the local Wikipedia dense retrieval service with a natural-language query. "
                "Always retrieves 3 passages; the model cannot set k or topk."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
}


def tool_schema(name: str) -> dict[str, Any]:
    return deepcopy(_SCHEMAS[name])


def tool_schemas() -> list[dict[str, Any]]:
    return [tool_schema(name) for name in _SCHEMAS]


def validate_parameters(parameters: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(parameters, dict):
        raise ValueError(f"{path} must be an object")
    if expected_type == "string" and not isinstance(parameters, str):
        raise ValueError(f"{path} must be a string")
    if "enum" in schema and parameters not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if expected_type == "string" and schema.get("minLength") and not parameters.strip():
        raise ValueError(f"{path} must be non-empty")
    if "oneOf" in schema:
        matches = 0
        for alternative in schema["oneOf"]:
            try:
                validate_parameters(parameters, alternative, path)
            except ValueError:
                continue
            matches += 1
        if matches != 1:
            raise ValueError(
                f'{path} must be {{"type":"corpus"}} or {{"type":"candidates","candidate_set_id":"<handle>"}}'
            )
        return
    if expected_type == "object":
        for required in schema.get("required", []):
            if required not in parameters:
                raise ValueError(f"missing {path}.{required}")
        properties = schema.get("properties", {})
        for name, value in parameters.items():
            if name not in properties:
                if schema.get("additionalProperties") is False:
                    raise ValueError(f"unexpected {path}.{name}")
                continue
            validate_parameters(value, properties[name], f"{path}.{name}")
