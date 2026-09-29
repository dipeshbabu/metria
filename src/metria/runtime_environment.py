"""Pinned local Python environments and bounded private worker messages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .processes import run_process
from .recipes import _json_value, _unique_json_object
from .runtimes.llamacpp import _sha256_file
from .runtimes.vllm_artifacts import installed_runtime_identity

ENVIRONMENT_SCHEMA = "metria.runtime_environment.v1"
REQUEST_SCHEMA = "metria.runtime_worker_request.v1"
RECEIPT_SCHEMA = "metria.runtime_worker_receipt.v1"


def source_digest() -> str:
    from .verification_trials import _implementation_digest

    return _implementation_digest()


def pin(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("runtime environment pins must be lowercase SHA256 digests")
    return value


def python_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    # Resolve the directory, retaining the final interpreter symlink so Python
    # still discovers this virtual environment's pyvenv.cfg.
    return value.parent.resolve() / value.name


def current_environment() -> dict[str, Any]:
    executable = python_path(sys.executable)
    prefix = Path(sys.prefix).resolve()
    library = prefix / "lib/libiomp5.so"
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "python": str(executable),
        "python_sha256": _sha256_file(executable),
        "prefix": str(prefix),
        "metria_sha256": source_digest(),
        "runtime": installed_runtime_identity(),
        "openmp": {"path": str(library), "sha256": _sha256_file(library)}
        if library.is_file()
        else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def validate_environment(value: Any, *, metria_sha256: str) -> Mapping[str, Any]:
    fields = {
        "schema",
        "python",
        "python_sha256",
        "prefix",
        "metria_sha256",
        "runtime",
        "openmp",
        "cpu_affinity",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value["schema"] != ENVIRONMENT_SCHEMA
    ):
        raise ValueError("a complete pinned runtime environment descriptor is required")
    if pin(value["metria_sha256"]) != metria_sha256:
        raise ValueError(
            "install the same Metria wheel in the controller and both runtime environments"
        )
    pin(value["python_sha256"])
    for name in ("python", "prefix"):
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute():
            raise ValueError("runtime interpreter and prefix must be absolute paths")
    runtime = value["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("status") != "verified"
        or not isinstance(runtime.get("version"), str)
    ):
        raise ValueError("runtime content identity is missing")
    pin(runtime.get("sha256"))
    if value["openmp"] is not None:
        library = value["openmp"]
        if (
            not isinstance(library, Mapping)
            or set(library) != {"path", "sha256"}
            or not isinstance(library["path"], str)
            or not Path(library["path"]).is_absolute()
        ):
            raise ValueError(
                "preloaded OpenMP library requires an absolute path and pin"
            )
        pin(library["sha256"])
    affinity = value["cpu_affinity"]
    if (
        not isinstance(affinity, (list, tuple))
        or not affinity
        or any(type(core) is not int or core < 0 for core in affinity)
        or list(affinity) != sorted(set(affinity))
    ):
        raise ValueError("a supported Linux CPU affinity must be observed")
    return value


def child_environment(
    python: Path,
    *,
    descriptor: Mapping[str, Any] | None = None,
    binding: str | None = None,
) -> dict[str, str]:
    result = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "VLLM_CPU_OMP_THREADS_BIND",
        "VLLM_CPU_KVCACHE_SPACE",
        "OMP_NUM_THREADS",
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
    ):
        result.pop(name, None)
    result.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    library = python.parent.parent / "lib/libiomp5.so"
    if descriptor is None:
        if library.is_file():
            result["LD_PRELOAD"] = str(library)
    elif descriptor["openmp"] is not None:
        library = Path(descriptor["openmp"]["path"])
        if _sha256_file(library) != descriptor["openmp"]["sha256"]:
            raise ValueError("preloaded library changed after environment preparation")
        result["LD_PRELOAD"] = str(library)
    if binding is not None:
        result.update(
            {
                "OMP_NUM_THREADS": "2",
                "VLLM_CPU_KVCACHE_SPACE": "1",
                "VLLM_CPU_OMP_THREADS_BIND": binding,
            }
        )
    return result


def write_private(path: Path, value: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(
            _json_value(value, path="runtime_worker"), sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError("worker JSON requires finite numbers")


def read_packet(path: Path, limit: int = 32 * 1024 * 1024) -> dict[str, Any]:
    return read_bound_packet(path, limit)[0]


def read_bound_packet(
    path: Path, limit: int = 32 * 1024 * 1024
) -> tuple[dict[str, Any], str]:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("runtime worker output exceeds its size limit")
    result = json.loads(
        raw, object_pairs_hook=_unique_json_object, parse_constant=_reject_constant
    )
    if not isinstance(result, dict):
        raise ValueError("runtime worker message must be an object")
    return result, hashlib.sha256(raw).hexdigest()


def validate_receipt(
    value: Mapping[str, Any], *, nonce: str, request_sha256: str
) -> None:
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("nonce") != nonce
        or value.get("request_sha256") != request_sha256
    ):
        raise ValueError("runtime worker receipt does not match this request")


def inspect_environment(python: str | Path, *, timeout: float = 120) -> dict[str, Any]:
    executable = python_path(python)
    expected_python = _sha256_file(executable)
    code = source_digest()
    nonce = secrets.token_hex(24)
    with tempfile.TemporaryDirectory(prefix="metria-environment-") as directory:
        root = Path(directory).resolve()
        if root.parent != Path(tempfile.gettempdir()).resolve():
            raise ValueError(
                "environment inspection directory escaped the temporary root"
            )
        request = root / "request.json"
        digest = write_private(
            request,
            {
                "schema": REQUEST_SCHEMA,
                "mode": "inspect",
                "nonce": nonce,
                "metria_sha256": code,
            },
        )
        result = run_process(
            [str(executable), "-I", "-m", "metria.runtime_worker", str(request)],
            cwd=root,
            env=child_environment(executable),
            timeout_s=timeout,
            max_output_bytes=2 * 1024 * 1024,
        )
        if result.timed_out or result.returncode != 0:
            raise ValueError(
                "runtime inspection failed; install the same Metria wheel and required vLLM dependencies in that environment"
            )
        receipt = read_packet(root / "receipt.json")
        validate_receipt(receipt, nonce=nonce, request_sha256=digest)
        observed = validate_environment(receipt.get("environment"), metria_sha256=code)
        if (
            observed["python"] != str(executable)
            or observed["python_sha256"] != expected_python
        ):
            raise ValueError("the requested interpreter was not independently observed")
    return dict(observed)
