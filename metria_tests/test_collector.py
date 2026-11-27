"""Host evidence collector: facts, redaction, signing and the HTTP endpoint."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import pytest

from metria.cli import main
from metria.collector import (
    CollectorConfig,
    collect_envelope,
    collect_facts,
    make_handler,
    model_inventory,
    parse_vllm_command,
)
from metria.evidence_envelope import (
    Challenge,
    KeyRing,
    TrustedKey,
    TrustPolicy,
    verify_envelope,
)

KEY = TrustedKey("host-a", "hmac-sha256", b"c" * 32)
CHALLENGE = Challenge("nonce-0123456789abcdef", "run-7", "candidate", "fp8-pool")
SMI = "0, NVIDIA L4, 8.9, 23034, 595.91.07, GPU-aaaa\n1, NVIDIA L4, 8.9, 23034, 595.91.07, GPU-bbbb\n"


def _fake_proc(tmp_path, model_dir, environ):
    proc = tmp_path / "proc" / "4242"
    proc.mkdir(parents=True)
    argv = [
        "/opt/venv/bin/python3", "-m", "vllm.entrypoints.cli.main", "serve",
        str(model_dir), "--kv-cache-dtype=fp8", "--tensor-parallel-size", "2",
        "--dtype", "float16", "--enable-prefix-caching", "--max-model-len", "512",
    ]  # fmt: skip
    (proc / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (proc / "environ").write_bytes(
        b"\0".join(f"{k}={v}".encode() for k, v in environ.items()) + b"\0"
    )
    return tmp_path / "proc"


@pytest.fixture
def model_dir(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"a": 1}')
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "README.md").write_text("ignored")
    return root


def test_parse_vllm_command_reads_engine_flags():
    facts, model = parse_vllm_command(
        ["vllm", "serve", "org/model", "-tp", "2", "--quantization=awq",
         "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.5"]
    )  # fmt: skip
    assert model == "org/model"
    assert facts == {
        "engine.tensor_parallel_size": 2,
        "engine.quantization": "awq",
        "engine.enable_prefix_caching": False,
        "engine.gpu_memory_utilization": 0.5,
    }
    _, model = parse_vllm_command(["python", "-m", "vllm.x", "--model", "/m"])
    assert model == "/m"


def test_model_inventory_hashes_model_files_only(model_dir):
    inventory = model_inventory(model_dir)
    assert inventory == {
        "config.json": hashlib.sha256(b'{"a": 1}').hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
    }


def test_collect_facts_from_process_gpus_and_model(tmp_path, model_dir):
    proc = _fake_proc(
        tmp_path,
        model_dir,
        {"NCCL_P2P_DISABLE": "1", "HF_TOKEN": "secret", "VLLM_API_KEY": "secret"},
    )
    facts, unavailable = collect_facts(
        CollectorConfig(
            pid=4242,
            proc_root=proc,
            run=lambda argv: SMI,
            image_digest="sha256:" + "d" * 64,
        )
    )
    assert facts["engine.kv_cache_dtype"] == "fp8"
    assert facts["engine.tensor_parallel_size"] == 2
    assert facts["engine.enable_prefix_caching"] is True
    assert facts["engine.command"] == "vllm"
    assert facts["environment.numerics"] == {"NCCL_P2P_DISABLE": "1"}
    assert set(facts["model.files"]) == {"config.json", "model.safetensors"}
    assert facts["gpu.driver_version"] == "595.91.07"
    assert [d["index"] for d in facts["gpu.devices"]] == [0, 1]
    assert "GPU-aaaa" not in json.dumps(facts)
    assert str(model_dir) not in json.dumps(facts)
    assert unavailable == []


def test_unreadable_sources_are_unavailable_not_guessed(tmp_path):
    def broken(argv):
        raise FileNotFoundError("nvidia-smi")

    facts, unavailable = collect_facts(
        CollectorConfig(pid=999, proc_root=tmp_path, run=broken)
    )
    assert facts == {}
    assert set(unavailable) == {
        "engine.command",
        "model.files",
        "model.path_sha256",
        "gpu.devices",
        "environment.numerics",
        "image.digest",
    }


def test_envelope_verifies_with_the_trusted_key(tmp_path, model_dir):
    proc = _fake_proc(tmp_path, model_dir, {})
    now = datetime(2026, 11, 27, tzinfo=timezone.utc)
    envelope = collect_envelope(
        CollectorConfig(pid=4242, proc_root=proc, run=lambda argv: SMI),
        KEY,
        CHALLENGE.as_mapping(),
        now=now,
    )
    verified = verify_envelope(
        envelope,
        keyring=KeyRing({KEY.key_id: KEY}),
        challenge=CHALLENGE,
        policy=TrustPolicy(
            require_collector=True, required_facts=("engine.kv_cache_dtype",)
        ),
        now=now,
    )
    assert verified["facts"]["engine.kv_cache_dtype"] == "fp8"


def test_http_endpoint_answers_challenges_and_rejects_bad_ones(tmp_path, model_dir):
    config = CollectorConfig(model_root=model_dir, query_gpus=False)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, KEY))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = json.dumps(CHALLENGE.as_mapping()).encode()
        with urllib.request.urlopen(
            urllib.request.Request(url + "/evidence", data=body)
        ) as r:
            envelope = json.loads(r.read())
        assert envelope["payload"]["challenge"] == CHALLENGE.as_mapping()
        assert envelope["key_id"] == "host-a"
        bad = json.dumps({"nonce": "short"}).encode()
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url + "/evidence", data=bad))
        assert error.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url + "/other", data=body))
        assert error.value.code == 404
    finally:
        server.shutdown()


def test_cli_keygen_prints_a_32_byte_key():
    out = io.StringIO()
    assert main(["collector", "keygen"], stdout=out) == 0
    assert len(base64.b64decode(out.getvalue().strip())) == 32


def test_key_from_environment(monkeypatch):
    from metria.collector import key_from_environment

    monkeypatch.setenv("METRIA_COLLECTOR_KEY", base64.b64encode(b"z" * 32).decode())
    assert key_from_environment("METRIA_COLLECTOR_KEY", "k1").material == b"z" * 32
    monkeypatch.setenv("METRIA_COLLECTOR_KEY", "***")
    with pytest.raises(ValueError, match="base64"):
        key_from_environment("METRIA_COLLECTOR_KEY", "k1")
    monkeypatch.delenv("METRIA_COLLECTOR_KEY")
    with pytest.raises(ValueError, match="not set"):
        key_from_environment("METRIA_COLLECTOR_KEY", "k1")
