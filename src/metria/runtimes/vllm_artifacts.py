"""Content identity for local model payloads and installed vLLM wheel files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

_PAYLOAD_SUFFIXES = frozenset(
    {".json", ".txt", ".model", ".safetensors", ".bin", ".tiktoken", ".jinja"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(rows: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _hub_tree_metadata(path: PurePath) -> bool:
    """Recognize only Hub local-directory tree-cache receipts, not payloads."""
    return (
        len(path.parts) == 4
        and path.parts[:3] == (".cache", "huggingface", "trees")
        and path.suffix == ".json"
        and len(path.stem) == 40
        and all(char in "0123456789abcdef" for char in path.stem)
    )


def validate_file_pins(files: Any) -> dict[str, str]:
    if not isinstance(files, Mapping) or not files or len(files) > 10_000:
        raise ValueError("model.files must contain a bounded file-to-SHA256 manifest")
    result = {}
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or ":" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
        ):
            raise ValueError("model file names must be portable relative paths")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or path.suffix not in _PAYLOAD_SUFFIXES
            or _hub_tree_metadata(path)
        ):
            raise ValueError(
                "model manifest contains an unsafe or unsupported payload path"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("model file pins must be lowercase SHA256 digests")
        result[name] = digest
    if not {"config.json", "tokenizer_config.json"} <= result.keys():
        raise ValueError("model manifest must pin model and tokenizer configuration")
    if not ({"tokenizer.json", "tokenizer.model"} & result.keys()):
        raise ValueError("model manifest must pin tokenizer content")
    if not any(name.endswith((".safetensors", ".bin")) for name in result):
        raise ValueError("model manifest must pin model weights")
    return result


def verify_model_files(directory: str | Path, files: Any) -> dict[str, Any]:
    """Hash each declared payload; HF snapshot file symlinks are supported."""
    pins = validate_file_pins(files)
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("verification requires an existing local model directory")
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in _PAYLOAD_SUFFIXES
        and not _hub_tree_metadata(path.relative_to(root))
    }
    if present != set(pins):
        raise ValueError("model payload inventory differs from its pinned manifest")
    rows = []
    for name, expected in sorted(pins.items()):
        path = root / name
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                "model or tokenizer content differs from its trusted file pin"
            )
        rows.append(
            {"path": name, "sha256": observed, "size_bytes": path.stat().st_size}
        )
    return {
        "status": "verified",
        "sha256": _digest(rows),
        "files": rows,
        "source": "hashed_local_model_and_tokenizer_payloads",
    }


def installed_runtime_identity(*, loaded_module: Any | None = None) -> dict[str, Any]:
    """Hash installed vLLM content without importing or loading an inference engine."""
    distribution = importlib.metadata.distribution("vllm")
    paths = distribution.files
    if paths is None:
        raise ValueError("vLLM installation has no wheel file inventory")
    if loaded_module is not None:
        loaded = getattr(loaded_module, "__file__", None)
        expected = distribution.locate_file("vllm/__init__.py")
        if (
            not isinstance(loaded, str)
            or Path(loaded).resolve() != Path(str(expected)).resolve()
        ):
            raise ValueError(
                "loaded vLLM module does not match the pinned installation"
            )
    rows = []
    for relative in sorted(paths, key=str):
        name = relative.as_posix()
        if (
            not name.startswith(("vllm/", "vllm.libs/"))
            or "__pycache__" in relative.parts
            or name.endswith(".pyc")
        ):
            continue
        path = Path(str(distribution.locate_file(relative)))
        if not path.is_file():
            raise ValueError("installed vLLM wheel file is missing")
        rows.append(
            {"path": name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    if not rows:
        raise ValueError("vLLM wheel inventory contains no runtime payload")
    dependencies = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "tokenizers")
    }
    identity = {
        "version": distribution.version,
        "files": rows,
        "dependencies": dependencies,
    }
    return {
        "status": "verified",
        "sha256": _digest(identity),
        "version": distribution.version,
        "dependencies": dependencies,
        "file_count": len(rows),
        "source": "hashed_installed_vllm_wheel_and_dependency_versions",
    }


def require_runtime_pin(
    environment: Mapping[str, Any], *, loaded_module: Any | None = None
) -> dict[str, Any] | None:
    expected = environment.get("vllm_distribution_sha256")
    if expected is None:
        return None
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise ValueError("vllm_distribution_sha256 must be a lowercase SHA256 digest")
    observed = installed_runtime_identity(loaded_module=loaded_module)
    if observed["sha256"] != expected:
        raise ValueError(
            "installed vLLM content or dependency versions differ from the runtime pin"
        )
    return observed
