from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass


OUTPUT_TRUNCATION_MARKER = "\n[... output truncated at source byte limit]"


@dataclass
class PipelineResult:
    stdout: str | bytes
    stderr: str | bytes
    returncode: int
    truncated: bool = False


def bound_output(contents: str) -> str:
    limit = int(os.environ.get("GREPSEEK_TOOL_MAX_OUTPUT_BYTES", "0"))
    if limit <= 0 or len(contents) <= limit // 4:
        return contents
    encoded = contents.encode("utf-8")
    if len(encoded) <= limit:
        return contents
    return encoded[:limit].decode("utf-8", errors="ignore") + OUTPUT_TRUNCATION_MARKER


def run_pipeline(
    command: str | list[str],
    *,
    timeout: float,
    env: dict | None = None,
    cwd: str | None = None,
    input: str | bytes | None = None,
    text: bool = True,
    max_output_bytes: int | None = None,
) -> PipelineResult:
    if max_output_bytes is None:
        max_output_bytes = int(os.environ.get("GREPSEEK_TOOL_MAX_OUTPUT_BYTES", "0"))
    if max_output_bytes < 0:
        raise ValueError("max_output_bytes must be nonnegative")
    input_bytes = input.encode("utf-8") if isinstance(input, str) else input
    process = subprocess.Popen(
        command,
        shell=isinstance(command, str),
        executable="/bin/bash" if isinstance(command, str) else None,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    input_offset = 0
    truncated = False

    def kill_group():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        with selectors.DefaultSelector() as selector:
            for name in outputs:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            if input_bytes:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            elif process.stdin is not None:
                process.stdin.close()
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(min(remaining, 0.1)):
                    if key.data == "stdin":
                        try:
                            input_offset += os.write(key.fd, input_bytes[input_offset:input_offset + 65536])
                        except BrokenPipeError:
                            input_offset = len(input_bytes)
                        if input_offset == len(input_bytes):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    output = outputs[key.data]
                    if max_output_bytes:
                        available = max(0, max_output_bytes - len(output))
                        output.extend(chunk[:available])
                        if key.data == "stdout" and len(chunk) > available:
                            truncated = True
                            kill_group()
                    else:
                        output.extend(chunk)
            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
    finally:
        kill_group()
        process.wait()
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name)
            if stream is not None:
                stream.close()
    stdout, stderr = bytes(outputs["stdout"]), bytes(outputs["stderr"])
    if text:
        stdout = stdout.decode("utf-8", errors="replace")
        stderr = stderr.decode("utf-8", errors="replace")
    return PipelineResult(stdout, stderr, 0 if truncated else returncode, truncated)
