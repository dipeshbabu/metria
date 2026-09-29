"""Bounded GGUF v3 inspection for F32/F16 to mixed Q8_0 qualification.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
Only metadata and tensor tables are read; model content pins bind the weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

_HEADER_LIMIT = 64 * 1024 * 1024
_ITEM_LIMIT = 1_000_000
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_STORAGE = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34)}
_QUANTIZATION_KEYS = {"general.file_type", "general.quantization_version"}
_RETAIN = _QUANTIZATION_KEYS | {
    "general.architecture",
    "general.alignment",
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
}


class _Reader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream = stream
        self.limit = min(size, _HEADER_LIMIT)
        self.items_left = _ITEM_LIMIT
        self.digest: Callable[[bytes], object] | None = None

    def take(self, count: int) -> bytes:
        if count < 0 or self.stream.tell() + count > self.limit:
            raise ValueError("GGUF header is truncated or exceeds the inspection limit")
        value = self.stream.read(count)
        if len(value) != count:
            raise ValueError("GGUF header changed or ended while reading")
        if self.digest is not None:
            self.digest(value)
        return value

    def number(self, format_: str) -> Any:
        return struct.unpack("<" + format_, self.take(struct.calcsize(format_)))[0]

    def text(self, maximum: int, *, keep: bool = True) -> str | None:
        size = self.number("Q")
        if size > maximum:
            raise ValueError("GGUF string exceeds the qualified inspection limit")
        if keep:
            return self.take(size).decode("utf-8", errors="strict")
        self.skip(size)
        return None

    def skip(self, count: int, *, boolean: bool = False) -> None:
        while count:
            chunk = self.take(min(count, 1024 * 1024))
            if boolean and any(value not in (0, 1) for value in chunk):
                raise ValueError("GGUF boolean values must be zero or one")
            count -= len(chunk)

    def value(self, kind: int, *, depth: int = 0, keep: bool = False) -> Any:
        if depth > 8:
            raise ValueError("GGUF metadata nesting exceeds the inspection limit")
        if kind in _SCALARS:
            if kind == 7:
                raw = self.number("B")
                if raw not in (0, 1):
                    raise ValueError("GGUF boolean values must be zero or one")
                value: Any = bool(raw)
            else:
                value = self.number(_SCALARS[kind])
            return value if keep else None
        if kind == 8:
            return self.text(16 * 1024 * 1024, keep=keep)
        if kind != 9:
            raise ValueError("unsupported GGUF metadata type")
        element_type, count = self.number("I"), self.number("Q")
        if count > self.items_left:
            raise ValueError("GGUF array exceeds the inspection limit")
        self.items_left -= count
        if element_type in _SCALARS:
            self.skip(
                count * struct.calcsize(_SCALARS[element_type]),
                boolean=element_type == 7,
            )
        elif element_type in (8, 9):
            for _ in range(count):
                self.value(element_type, depth=depth + 1)
        else:
            raise ValueError("unsupported GGUF array element type")
        return {"element_type": element_type, "count": count} if keep else None


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _metadata(reader: _Reader, count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 1 <= count <= 10_000:
        raise ValueError("GGUF metadata count is outside the qualified limit")
    retained: dict[str, Any] = {}
    fingerprints = {}
    for _ in range(count):
        name = reader.text(65535)
        if not name or not name.isascii() or "\0" in name or name in fingerprints:
            raise ValueError("GGUF metadata keys must be unique nonempty ASCII strings")
        if name.startswith("split."):
            raise ValueError("split GGUF models are outside this qualification scope")
        kind = reader.number("I")
        digest = hashlib.sha256()
        reader.digest = digest.update
        value = reader.value(kind, keep=name in _RETAIN)
        reader.digest = None
        fingerprints[name] = {"type": kind, "sha256": digest.hexdigest()}
        if name in _RETAIN:
            retained[name] = value
    return retained, fingerprints


def _tensors(reader: _Reader, count: int) -> list[dict[str, Any]]:
    if not 1 <= count <= 100_000:
        raise ValueError("GGUF tensor count is outside the qualified limit")
    tensors = []
    names = set()
    for _ in range(count):
        name = reader.text(64)
        dimensions = reader.number("I")
        if not name or "\0" in name or name in names or not 1 <= dimensions <= 4:
            raise ValueError("GGUF tensor name or dimensionality is invalid")
        names.add(name)
        shape = [reader.number("Q") for _ in range(dimensions)]
        elements = math.prod(shape)
        if not all(shape) or elements > 2**63:
            raise ValueError("GGUF tensor shape is invalid or too large")
        kind, offset = reader.number("I"), reader.number("Q")
        if kind not in _STORAGE:
            raise ValueError(
                "qualified GGUF inspection supports F32, F16 and Q8_0 storage only"
            )
        storage, block, size = _STORAGE[kind]
        if shape[0] % block:
            raise ValueError("GGUF tensor width does not match its storage block size")
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "storage": storage,
                "offset": offset,
                "bytes": elements // block * size,
            }
        )
    return tensors


def _validate_payload(
    tensors: list[dict[str, Any]], start: int, size: int, alignment: Any
) -> None:
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or not 1 <= alignment <= 4096
        or alignment & (alignment - 1)
    ):
        raise ValueError("GGUF alignment must be a supported positive power of two")
    data_start = (start + alignment - 1) // alignment * alignment
    end = 0
    for tensor in sorted(tensors, key=lambda item: item["offset"]):
        offset, length = tensor["offset"], tensor["bytes"]
        if offset % alignment or offset < end or data_start + offset + length > size:
            raise ValueError(
                "GGUF tensor data is misaligned, overlapping or out of bounds"
            )
        end = offset + length


def _identity(
    metadata: dict[str, Any],
    fingerprints: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> dict[str, Any]:
    architecture = metadata.get("general.architecture")
    tokenizer = metadata.get("tokenizer.ggml.model")
    tokens = metadata.get("tokenizer.ggml.tokens")
    if any(
        not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.isascii()
        for value in (architecture, tokenizer)
    ):
        raise ValueError(
            "GGUF model architecture and embedded tokenizer identity are required"
        )
    if (
        not isinstance(tokens, dict)
        or tokens != {"element_type": 8, "count": tokens.get("count")}
        or not tokens["count"]
    ):
        raise ValueError("GGUF tokenizer must contain a nonempty string token array")
    file_type = metadata.get("general.file_type")
    if file_type is not None and (
        isinstance(file_type, bool)
        or not isinstance(file_type, int)
        or file_type not in (0, 1, 7)
    ):
        raise ValueError(
            "qualified GGUF file types are F32, mostly F16 and mostly Q8_0"
        )
    counts = dict(sorted(Counter(tensor["storage"] for tensor in tensors).items()))
    if (file_type == 7) != (counts.get("Q8_0", 0) > 0):
        raise ValueError("GGUF quantization label does not match actual tensor storage")
    if file_type == 0 and set(counts) != {"F32"}:
        raise ValueError("GGUF F32 label does not match actual tensor storage")
    if file_type == 1 and not counts.get("F16"):
        raise ValueError("GGUF F16 label does not match actual tensor storage")
    if file_type == 7 and (
        type(metadata.get("general.quantization_version")) is not int
        or metadata["general.quantization_version"] != 2
    ):
        raise ValueError("qualified Q8_0 requires GGML quantization version 2")
    return {
        "schema": "metria.gguf_identity.v1",
        "format": "GGUFv3-little-endian",
        "architecture": architecture,
        "file_type": file_type,
        "quantization_version": metadata.get("general.quantization_version"),
        "vocab_size": tokens["count"],
        "tensor_count": len(tensors),
        "tensor_types": counts,
        "tokenizer_sha256": _fingerprint(
            {k: v for k, v in fingerprints.items() if k.startswith("tokenizer.")}
        ),
        "controls_sha256": _fingerprint(
            {k: v for k, v in fingerprints.items() if k not in _QUANTIZATION_KEYS}
        ),
        "tensor_layout_sha256": _fingerprint(
            sorted(
                ({"name": t["name"], "shape": t["shape"]} for t in tensors),
                key=lambda t: t["name"],
            )
        ),
    }


def inspect_gguf(path: str | Path) -> dict[str, Any]:
    """Inspect a bounded single-file layout without loading tensor payloads."""
    source = Path(path)
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        reader = _Reader(stream, before.st_size)
        if reader.take(4) != b"GGUF" or reader.number("I") != 3:
            raise ValueError(
                "qualified inspection requires little-endian GGUF version 3"
            )
        tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
        metadata, fingerprints = _metadata(reader, metadata_count)
        tensors = _tensors(reader, tensor_count)
        _validate_payload(
            tensors,
            stream.tell(),
            before.st_size,
            metadata.get("general.alignment", 32),
        )
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError("GGUF file changed while metadata was inspected")
    return _identity(metadata, fingerprints, tensors)
