"""Aggregate executed hybrid tool calls, separately from reward statistics.

Call fractions divide summed counts (not per-trajectory fractions). Trajectory
fractions include rollouts with no calls. Failed executions count as calls;
malformed requests rejected before dispatch do not.
"""

from collections import defaultdict

import numpy as np

COUNT_PREFIX = "tool_usage/count/"
TOOLS = ("dense_search", "shell", "shell_corpus", "shell_candidates", "shell_unknown_scope")


def tool_usage_columns(non_tensor_batch):
    """Return aligned numeric columns; an absent tool_counts means unavailable."""
    counts = non_tensor_batch.get("tool_counts")
    if counts is None:
        return {}
    traces = non_tensor_batch.get("tool_trace", [None] * len(counts))
    columns = {COUNT_PREFIX + tool: [] for tool in TOOLS}
    for record, trace in zip(counts, traces, strict=True):
        record = record or {}
        row = {tool: int(record.get(tool, 0)) for tool in ("dense_search", "shell")}
        scopes = defaultdict(int)
        for event in trace or []:
            if event.get("tool_name") == "shell":
                scope = event.get("scope")
                if isinstance(scope, dict):
                    scope = scope.get("type")
                scopes[scope if scope in ("corpus", "candidates") else "unknown_scope"] += 1
        row.update({"shell_" + scope: scopes[scope] for scope in ("corpus", "candidates")})
        row["shell_unknown_scope"] = row["shell"] - row["shell_corpus"] - row["shell_candidates"]
        if row["shell_unknown_scope"] < 0:
            raise ValueError("Shell scope counts exceed executed shell calls")
        for tool in TOOLS:
            columns[COUNT_PREFIX + tool].append(row[tool])
    return columns


def extend_tool_usage(destination, columns):
    for key, values in columns.items():
        destination.setdefault(key, []).extend(values)


def summarize_tool_usage(columns, prefix):
    if COUNT_PREFIX + "dense_search" not in columns:
        return {}
    arrays = {tool: np.asarray(columns[COUNT_PREFIX + tool], dtype=np.int64) for tool in TOOLS}
    n = len(arrays["dense_search"])
    if not n:
        return {}
    if any(len(values) != n for values in arrays.values()):
        raise ValueError("Tool usage columns must have the same number of trajectories")
    total = int((arrays["dense_search"] + arrays["shell"]).sum())
    metrics = {
        f"{prefix}/trajectories": n,
        f"{prefix}/calls/total": total,
        f"{prefix}/calls_per_trajectory": total / n,
    }
    for tool, values in arrays.items():
        count = int(values.sum())
        metrics[f"{prefix}/calls/{tool}"] = count
        metrics[f"{prefix}/call_fraction/{tool}"] = count / total if total else 0.0
        metrics[f"{prefix}/trajectory_fraction/{tool}"] = float(np.mean(values > 0))
    dense, shell = arrays["dense_search"] > 0, arrays["shell"] > 0
    for name, mask in (
        ("mixed", dense & shell),
        ("dense_only", dense & ~shell),
        ("shell_only", shell & ~dense),
        ("no_tool", ~dense & ~shell),
    ):
        metrics[f"{prefix}/trajectory_fraction/{name}"] = float(np.mean(mask))
    return metrics
