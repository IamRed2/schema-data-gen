"""Run a private llama.cpp worker for the lifetime of one CLI invocation."""

import json
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from schema_generator.value_generator import INSTRUCTIONS, decode_value, value_output_schema


@contextmanager
def load_server_backend(executable, model, adapter=None, device="auto", *, format_messages=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    api_key = secrets.token_urlsafe(24)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{port}"
    command = [
        str(executable), "--model", str(model), "--host", "127.0.0.1",
        "--port", str(port), "--api-key", api_key, "--ctx-size", "4096",
        "--parallel", "1", "--reasoning", "off", "--no-warmup",
    ]
    if adapter is not None:
        command.extend(["--lora", str(adapter)])
    if device != "auto":
        command.extend(["--gpu-layers", "999" if device == "gpu" else "0"])
    if device == "gpu":
        command.append("--verbose")

    def call(path, payload=None, timeout=90):
        request = urllib.request.Request(
            url + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)

    # Windows can briefly retain a worker's log handle after the process exits.
    with tempfile.TemporaryDirectory(prefix="schema-llama-", ignore_cleanup_errors=True) as folder:
        log_path = Path(folder) / "worker.log"
        with log_path.open("wb") as log:
            started = time.perf_counter()
            process = subprocess.Popen(
                command, cwd=executable.parent, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                while process.poll() is None and time.perf_counter() - started < 120:
                    try:
                        call("/v1/models", timeout=1)
                        break
                    except (OSError, ValueError):
                        time.sleep(0.1)
                else:
                    detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    raise RuntimeError(f"llama-server did not become ready: {detail}")
                messages_log = log_path.read_text(encoding="utf-8", errors="replace")
                offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", messages_log)
                if device == "gpu" and not any(int(layers) > 0 for layers, _ in offload):
                    raise RuntimeError("GPU was requested but llama-server did not report GPU layer offload.")
                placement = f", GPU layers {offload[-1][0]}/{offload[-1][1]}" if offload else ""
                print(f"llama-server ready in {time.perf_counter() - started:.2f}s{placement}", file=sys.stderr)

                def generate(request, seed):
                    messages = [
                        {"role": "system", "content": INSTRUCTIONS},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False) + "\n/no_think"},
                    ] if format_messages is None else format_messages(request)
                    payload = {
                        "messages": messages,
                        "response_format": {"type": "json_schema", "schema": value_output_schema(request)},
                        "max_tokens": 512, "seed": seed, "cache_prompt": True,
                    }
                    try:
                        result = call("/v1/chat/completions", payload)
                    except urllib.error.HTTPError as error:
                        detail = error.read(2000).decode("utf-8", errors="replace")
                        raise RuntimeError(f"llama-server returned HTTP {error.code}: {detail}") from error
                    return decode_value(result["choices"][0]["message"]["content"])

                yield generate
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
