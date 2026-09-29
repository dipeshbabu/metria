from __future__ import annotations

import struct

import pytest

from metria.runtimes.gguf_identity import inspect_gguf


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _model_bytes(
    *,
    quantized=False,
    tokens=("a", "b"),
    extra=(),
    offset=0,
    shape=(32, 2),
    version=3,
    metadata_override=None,
    omit=(),
):
    metadata = {
        "general.architecture": (8, _string("llama")),
        "general.file_type": (4, struct.pack("<I", 7 if quantized else 0)),
        "tokenizer.ggml.model": (8, _string("llama")),
        "tokenizer.ggml.tokens": (
            9,
            struct.pack("<IQ", 8, len(tokens))
            + b"".join(_string(token) for token in tokens),
        ),
        "llama.embedding_length": (4, struct.pack("<I", 32)),
    }
    if quantized:
        metadata["general.quantization_version"] = (4, struct.pack("<I", 2))
    metadata.update(dict(extra))
    for key in omit:
        metadata.pop(key)
    if metadata_override is not None:
        metadata = metadata_override
    header = b"GGUF" + struct.pack("<IQQ", version, 1, len(metadata))
    for key, (kind, raw) in metadata.items():
        header += _string(key) + struct.pack("<I", kind) + raw
    header += _string("weight") + struct.pack("<I", len(shape))
    header += b"".join(struct.pack("<Q", value) for value in shape)
    header += struct.pack("<IQ", 8 if quantized else 0, offset)
    payload_size = 68 if quantized else 256
    return header + bytes((-len(header)) % 32) + bytes(payload_size)


def _write(tmp_path, data, name="model.gguf"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_real_tensor_storage_is_distinct_from_shared_model_tokenizer_controls(tmp_path):
    reference = inspect_gguf(_write(tmp_path, _model_bytes(), "reference.gguf"))
    candidate = inspect_gguf(
        _write(tmp_path, _model_bytes(quantized=True), "candidate.gguf")
    )
    assert reference["tensor_types"] == {"F32": 1}
    assert candidate["tensor_types"] == {"Q8_0": 1}
    assert reference["file_type"] == 0 and candidate["file_type"] == 7
    assert candidate["quantization_version"] == 2
    assert candidate["vocab_size"] == 2
    for key in ("tokenizer_sha256", "controls_sha256", "tensor_layout_sha256"):
        assert reference[key] == candidate[key]


def test_tokenizer_and_architecture_changes_are_not_hidden_by_quantization(tmp_path):
    baseline = inspect_gguf(_write(tmp_path, _model_bytes(), "baseline.gguf"))
    tokenizer = inspect_gguf(
        _write(tmp_path, _model_bytes(tokens=("a", "changed")), "tokenizer.gguf")
    )
    architecture = inspect_gguf(
        _write(
            tmp_path,
            _model_bytes(
                extra=[("llama.embedding_length", (4, struct.pack("<I", 64)))]
            ),
            "architecture.gguf",
        )
    )
    assert baseline["tokenizer_sha256"] != tokenizer["tokenizer_sha256"]
    assert baseline["controls_sha256"] != tokenizer["controls_sha256"]
    assert baseline["controls_sha256"] != architecture["controls_sha256"]
    assert baseline["tokenizer_sha256"] == architecture["tokenizer_sha256"]


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"notGGUF",
        _model_bytes(version=2),
        _model_bytes()[:-1],
        _model_bytes(offset=1),
        _model_bytes(offset=32),
        _model_bytes(shape=()),
        _model_bytes(shape=(0,)),
        _model_bytes(shape=(2**63, 2)),
        _model_bytes(quantized=True, shape=(31, 2)),
        _model_bytes(extra=[("split.count", (4, struct.pack("<I", 2)))]),
        _model_bytes(extra=[("general.alignment", (4, struct.pack("<I", 3)))]),
        _model_bytes(extra=[("general.file_type", (4, struct.pack("<I", 7)))]),
        _model_bytes(
            quantized=True, extra=[("general.file_type", (4, struct.pack("<I", 0)))]
        ),
        _model_bytes(
            quantized=True,
            extra=[("general.quantization_version", (4, struct.pack("<I", 99)))],
        ),
        _model_bytes(extra=[("tokenizer.ggml.tokens", (9, struct.pack("<IQ", 8, 0)))]),
        _model_bytes(extra=[("tokenizer.ggml.model", (8, _string("")))]),
        _model_bytes(extra=[("bad.boolean", (7, b"\x02"))]),
        _model_bytes(
            extra=[("bad.boolean_array", (9, struct.pack("<IQ", 7, 2) + b"\x00\x02"))]
        ),
        _model_bytes(extra=[("bad.type", (999, b""))]),
        _model_bytes(extra=[("bad.array", (9, struct.pack("<IQ", 999, 0)))]),
        _model_bytes(extra=[("bad.array", (9, struct.pack("<IQ", 4, 2**63)))]),
        _model_bytes(extra=[("bad.string", (8, struct.pack("<Q", 2**63)))]),
        b"GGUF" + struct.pack("<IQQ", 3, 1, 2**63),
        _model_bytes(metadata_override={}),
    ],
)
def test_malformed_or_mislabeled_artifacts_fail_before_inference(tmp_path, data):
    with pytest.raises((ValueError, UnicodeError)):
        inspect_gguf(_write(tmp_path, data))


def test_nested_arrays_are_hashed_but_deep_nesting_is_bounded(tmp_path):
    array = struct.pack("<IQI", 4, 1, 42)
    shallow = _model_bytes(
        extra=[("custom.array", (9, struct.pack("<IQ", 9, 1) + array))]
    )
    assert inspect_gguf(_write(tmp_path, shallow))["tensor_count"] == 1
    nested = array
    for _ in range(10):
        nested = struct.pack("<IQ", 9, 1) + nested
    with pytest.raises(ValueError, match="nesting"):
        inspect_gguf(
            _write(tmp_path, _model_bytes(extra=[("custom.array", (9, nested))]))
        )


def test_metadata_order_does_not_change_control_identity(tmp_path):
    extra = [
        ("custom.one", (4, struct.pack("<I", 1))),
        ("custom.two", (8, _string("two"))),
    ]
    left = inspect_gguf(_write(tmp_path, _model_bytes(extra=extra), "left.gguf"))
    right = inspect_gguf(
        _write(tmp_path, _model_bytes(extra=list(reversed(extra))), "right.gguf")
    )
    assert left == right


def test_absent_optional_file_type_is_not_invented(tmp_path):
    result = inspect_gguf(_write(tmp_path, _model_bytes(omit=("general.file_type",))))
    assert result["file_type"] is None
    assert result["tensor_types"] == {"F32": 1}


def test_duplicate_metadata_keys_are_not_silently_replaced(tmp_path):
    entry = _string("general.architecture") + struct.pack("<I", 8) + _string("llama")
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + entry + entry + bytes(256)
    with pytest.raises(ValueError, match="unique"):
        inspect_gguf(_write(tmp_path, data))


def test_overlapping_tensor_payloads_are_rejected(tmp_path):
    original = _model_bytes()
    start = original.index(_string("weight"))
    entry = _string("weight") + struct.pack("<IQQIQ", 2, 32, 2, 0, 0)
    other = _string("other") + struct.pack("<IQQIQ", 2, 32, 2, 0, 0)
    header = original[:8] + struct.pack("<Q", 2) + original[16:start] + entry + other
    data = header + bytes((-len(header)) % 32) + bytes(512)
    with pytest.raises(ValueError, match="overlapping"):
        inspect_gguf(_write(tmp_path, data))


def test_file_change_during_inspection_is_not_observed_identity(tmp_path, monkeypatch):
    import metria.runtimes.gguf_identity as module

    path = _write(tmp_path, _model_bytes())
    original = module._identity
    # Exercise the stat guard at the artifact boundary, before identity creation.
    original_tensors = module._tensors

    def mutate(reader, count):
        value = original_tensors(reader, count)
        path.write_bytes(path.read_bytes() + b"changed")
        return value

    monkeypatch.setattr(module, "_tensors", mutate)
    with pytest.raises(ValueError, match="changed while"):
        inspect_gguf(path)
    assert module._identity is original
