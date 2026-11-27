"""Optional host evidence collector for remote endpoint verification.

Run next to an inference server you operate. On request it reads facts the
server cannot be trusted to self-report and returns them in a signed,
challenge-bound ``metria.endpoint_evidence.v1`` envelope:

* engine configuration from the server process's own command line (vLLM);
* SHA-256 of every model and tokenizer file the server loaded;
* visible GPUs from ``nvidia-smi`` and the process's numerics environment
  (``NCCL_*``, ``VLLM_*``, CUDA and thread settings, secrets excluded);
* an optional container image digest supplied by the operator.

Facts that cannot be read are listed as unavailable, never guessed. Paths are
reported only as hashes. The collector holds an HMAC key; anyone with that key
can sign, so protect it like any credential.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

from .evidence_envelope import EnvelopeError, TrustedKey, sign
from .runtimes.numerics_environment import numerics_environment

COLLECTOR_VERSION = "1"
# vLLM serve flags recorded as engine facts (value-taking flags).
_VALUE_FLAGS = {
    "--dtype": "engine.dtype",
    "--kv-cache-dtype": "engine.kv_cache_dtype",
    "--tensor-parallel-size": "engine.tensor_parallel_size",
    "-tp": "engine.tensor_parallel_size",
    "--pipeline-parallel-size": "engine.pipeline_parallel_size",
    "-pp": "engine.pipeline_parallel_size",
    "--max-model-len": "engine.max_model_len",
    "--quantization": "engine.quantization",
    "--served-model-name": "engine.served_model_name",
    "--max-num-seqs": "engine.max_num_seqs",
    "--gpu-memory-utilization": "engine.gpu_memory_utilization",
    "--revision": "model.revision",
}
_BOOL_FLAGS = {
    "--enable-prefix-caching": ("engine.enable_prefix_caching", True),
    "--no-enable-prefix-caching": ("engine.enable_prefix_caching", False),
    "--enforce-eager": ("engine.enforce_eager", True),
}
_VLLM_DEFAULTS = {
    "engine.kv_cache_dtype": "auto",
    "engine.tensor_parallel_size": 1,
    "engine.pipeline_parallel_size": 1,
}
_MODEL_SUFFIXES = (
    ".json",
    ".safetensors",
    ".bin",
    ".model",
    ".txt",
    ".gguf",
    ".tiktoken",
)
_MAX_MODEL_FILES = 2000


def _number(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_vllm_command(argv: Sequence[str]) -> tuple[dict[str, Any], str | None]:
    """Engine facts and the model argument from a ``vllm serve`` command line."""

    args = list(argv)
    facts: dict[str, Any] = {}
    model: str | None = None
    if "serve" in args:
        position = args.index("serve")
        if position + 1 < len(args) and not args[position + 1].startswith("-"):
            model = args[position + 1]
    index = 0
    while index < len(args):
        item = args[index]
        name, has_inline, inline = item.partition("=")
        if name in _VALUE_FLAGS and (has_inline or index + 1 < len(args)):
            value = inline if has_inline else args[index + 1]
            facts[_VALUE_FLAGS[name]] = _number(value)
            index += 1 if has_inline else 2
            continue
        if name == "--model" and (has_inline or index + 1 < len(args)):
            model = inline if has_inline else args[index + 1]
            index += 1 if has_inline else 2
            continue
        if item in _BOOL_FLAGS:
            key, flag = _BOOL_FLAGS[item]
            facts[key] = flag
        index += 1
    return facts, model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_inventory(root: Path) -> dict[str, str]:
    """SHA-256 of model and tokenizer files, keyed by path relative to the model."""

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in _MODEL_SUFFIXES
    )
    if len(files) > _MAX_MODEL_FILES:
        raise ValueError("model directory has too many files to inventory")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in files}


def _read_proc(proc_root: Path, pid: int) -> tuple[list[str], dict[str, str]]:
    base = proc_root / str(pid)
    argv = [
        part.decode() for part in (base / "cmdline").read_bytes().split(b"\0") if part
    ]
    environ = {}
    for entry in (base / "environ").read_bytes().split(b"\0"):
        key, separator, value = entry.decode(errors="replace").partition("=")
        if separator:
            environ[key] = value
    return argv, environ


def _gpu_facts(run: Callable[[list[str]], str]) -> dict[str, Any]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    driver = None
    for line in output.strip().splitlines():
        index, name, cap, memory, driver, uuid = (
            part.strip() for part in line.split(",")
        )
        devices.append(
            {
                "index": int(index),
                "name": name,
                "compute_capability": cap,
                "memory_mib": int(float(memory)),
                "uuid_sha256": hashlib.sha256(uuid.encode()).hexdigest(),
            }
        )
    return {"gpu.devices": devices, "gpu.driver_version": driver}


def _default_run(command: list[str]) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        command, check=True, capture_output=True, text=True, timeout=30
    ).stdout


@dataclass
class CollectorConfig:
    """What to inspect; every source is optional."""

    pid: int | None = None
    proc_root: Path = Path("/proc")
    model_root: Path | None = None
    image_digest: str | None = None
    query_gpus: bool = True
    run: Callable[[list[str]], str] = field(default=_default_run, repr=False)


_KNOWN_FACTS = (
    "engine.command",
    "model.files",
    "model.path_sha256",
    "gpu.devices",
    "environment.numerics",
    "image.digest",
)


def collect_facts(config: CollectorConfig) -> tuple[dict[str, Any], list[str]]:
    """Gather what can be read; everything else is reported unavailable."""

    facts: dict[str, Any] = {}
    model_root = config.model_root
    if config.pid is not None:
        try:
            argv, environ = _read_proc(config.proc_root, config.pid)
        except OSError:
            argv, environ = [], {}
        if argv:
            engine, model = parse_vllm_command(argv)
            facts.update(engine)
            if any("vllm" in a for a in argv[:3]):
                # vLLM's documented defaults for omitted flags, listed explicitly.
                defaulted = [k for k in _VLLM_DEFAULTS if k not in engine]
                facts.update({k: _VLLM_DEFAULTS[k] for k in defaulted})
                facts["engine.defaulted"] = defaulted
            facts["engine.command"] = (
                "vllm" if any("vllm" in a for a in argv[:3]) else "unknown"
            )
            if model_root is None and model and Path(model).is_dir():
                model_root = Path(model)
            facts["environment.numerics"] = numerics_environment(environ)
    if model_root is not None and model_root.is_dir():
        facts["model.files"] = model_inventory(model_root)
        facts["model.path_sha256"] = hashlib.sha256(
            str(model_root.resolve()).encode()
        ).hexdigest()
    if config.query_gpus:
        try:
            facts.update(_gpu_facts(config.run))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if config.image_digest:
        facts["image.digest"] = config.image_digest
    unavailable = [name for name in _KNOWN_FACTS if name not in facts]
    return facts, unavailable


def collect_envelope(
    config: CollectorConfig,
    key: TrustedKey,
    challenge: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Collect facts and sign them, bound to the verifier's challenge."""

    facts, unavailable = collect_facts(config)
    payload = {
        "collector": {"name": "metria-collector", "version": COLLECTOR_VERSION},
        "challenge": dict(challenge),
        "collected_at": (now or datetime.now(timezone.utc)).isoformat(),
        "facts": facts,
        "unavailable": unavailable,
    }
    return sign(payload, key)


def key_from_environment(name: str, key_id: str) -> TrustedKey:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"environment variable {name} is not set")
    try:
        material = base64.b64decode(value, validate=True)
    except ValueError:
        raise ValueError(f"{name} must hold a base64 key") from None
    return TrustedKey(key_id, "hmac-sha256", material)


def make_handler(
    config: CollectorConfig, key: TrustedKey
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm = True

        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/evidence":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("challenge body size")
                challenge = json.loads(self.rfile.read(length))
                self._send(200, collect_envelope(config, key, challenge))
            except (ValueError, EnvelopeError):
                self._send(400, {"error": "invalid challenge"})

    return Handler


def add_collector_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "collector", help="serve signed host evidence for endpoint verification"
    )
    commands = parser.add_subparsers(dest="collector_command", required=True)
    serve = commands.add_parser("serve", help="answer evidence challenges over HTTP")
    keygen = commands.add_parser("keygen", help="print a new random base64 HMAC key")
    del keygen
    serve.add_argument("--pid", type=int, help="inference server process ID")
    serve.add_argument("--model-root", type=Path, help="model directory to inventory")
    serve.add_argument("--image-digest", help="container image digest, e.g. sha256:...")
    serve.add_argument("--key-env", default="METRIA_COLLECTOR_KEY")
    serve.add_argument("--key-id", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9105)
    serve.add_argument("--no-gpus", action="store_true", help="skip nvidia-smi")


def collector_command(args: Any, stdout: TextIO) -> int:
    if args.collector_command == "keygen":
        stdout.write(base64.b64encode(secrets.token_bytes(32)).decode() + "\n")
        return 0
    key = key_from_environment(args.key_env, args.key_id)
    config = CollectorConfig(
        pid=args.pid,
        model_root=args.model_root,
        image_digest=args.image_digest,
        query_gpus=not args.no_gpus,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(config, key))
    stdout.write(
        f"Serving signed evidence on http://{args.host}:{args.port}/evidence "
        f"(key {args.key_id}); stop with Ctrl+C\n"
    )
    stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
