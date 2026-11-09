"""Split shell pipelines without changing quoted or escaped argument text."""
from __future__ import annotations


def split_pipeline(command: str) -> list[str]:
    """Split only unquoted, unescaped pipes, preserving each stage verbatim.

    This is not a shell validator. Callers must still reject unsupported shell
    constructs and tokenize stages to detect malformed quoting.
    """
    stages: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            escaped = False
        elif character == "\\" and quote != "'":
            escaped = True
        elif quote is not None:
            if character == quote:
                quote = None
        elif character in ("'", '"'):
            quote = character
        elif character == "|":
            stages.append(command[start:index])
            start = index + 1
    stages.append(command[start:])
    return stages
