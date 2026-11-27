"""remote_endpoint_change.v1 with signed host collectors, end to end."""

from __future__ import annotations

import base64
import io
import json
import threading
from dataclasses import replace
from http.server import ThreadingHTTPServer

import pytest
from metria_tests.test_endpoint_verify import POLICY, PROMPTS, _verify, servers  # noqa: F401

from metria.cli import main
from metria.collector import CollectorConfig, make_handler
from metria.evidence_envelope import TrustedKey
from metria.preparation_endpoint import prepare_endpoint_recipe
from metria.verification_endpoint import validate_endpoint_recipe

KEY = TrustedKey("hosts-2026-11", "hmac-sha256", b"h" * 32)
ROGUE = TrustedKey("hosts-2026-11", "hmac-sha256", b"r" * 32)
SMI = "0, NVIDIA L4, 8.9, 23034, 595.91.07, GPU-{role}\n"


def _proc(tmp_path, role, kv):
    root = tmp_path / f"proc-{role}"
    pid_dir = root / "100"
    pid_dir.mkdir(parents=True)
    argv = ["python3", "-m", "vllm.entrypoints.cli.main", "serve", "/srv/model"]
    if kv != "auto":
        argv += ["--kv-cache-dtype", kv]
    (pid_dir / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    env = {
        "CUDA_VISIBLE_DEVICES": "0" if role == "reference" else "1",
        "NCCL_P2P_DISABLE": "1",
    }
    (pid_dir / "environ").write_bytes(
        b"\0".join(f"{k}={v}".encode() for k, v in env.items()) + b"\0"
    )
    return root


class Collectors:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.servers = []
        self.kv = {"reference": "auto", "candidate": "fp8"}
        self.keys = {"reference": KEY, "candidate": KEY}
        self.drivers = {"reference": ["595.91.07"], "candidate": ["595.91.07"]}
        ring = {
            KEY.key_id: {
                "algorithm": "hmac-sha256",
                "key": base64.b64encode(KEY.material).decode(),
            }
        }
        monkeypatch.setenv("COLLECTOR_KEYS", json.dumps(ring))

    def start(self):
        for role, env in (
            ("reference", "REF_COLLECTOR"),
            ("candidate", "CAND_COLLECTOR"),
        ):
            drivers = self.drivers[role]

            def smi(argv, role=role, drivers=drivers):
                driver = drivers.pop(0) if len(drivers) > 1 else drivers[0]
                return SMI.format(role=role).replace("595.91.07", driver)

            config = CollectorConfig(
                pid=100, proc_root=_proc(self.tmp_path, role, self.kv[role]), run=smi
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(config, self.keys[role])
            )
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)
            self.monkeypatch.setenv(env, f"http://127.0.0.1:{server.server_address[1]}")

    def close(self):
        for server in self.servers:
            server.shutdown()


@pytest.fixture
def collectors(servers, tmp_path, monkeypatch):  # noqa: F811
    instance = Collectors(tmp_path, monkeypatch)
    yield instance
    instance.close()


def _recipe(tmp_path, trust=None, declared=("auto", "fp8")):
    return prepare_endpoint_recipe(
        model_id="served-model",
        tokenizer_dir=tmp_path / "tok",
        prompts=PROMPTS,
        reference_url_env="REF_URL",
        candidate_url_env="CAND_URL",
        reference_declared={"kv_cache_dtype": declared[0]},
        candidate_declared={"kv_cache_dtype": declared[1]},
        policy=POLICY,
        max_tokens=3,
        collector={
            "reference_url_env": "REF_COLLECTOR",
            "candidate_url_env": "CAND_COLLECTOR",
            "keys_env": "COLLECTOR_KEYS",
            "reference_deployment": "auto-pool",
            "candidate_deployment": "fp8-pool",
            "trust": trust or {"max_age_s": 300},
        },
    )


def test_collector_observed_change_passes(collectors, tmp_path):
    collectors.start()
    result = _verify(_recipe(tmp_path), tmp_path / "out")
    manifest = result.manifest
    assert manifest["verdict"] == "PASS", json.dumps(manifest, default=str)[:3000]
    assert manifest["change"]["identity_authority"] == "collector_observed"
    assert (
        manifest["records"]["candidate"]["observed"]["identity_authority"]
        == "collector_observed"
    )
    assert (
        "observed by signed host collectors"
        in (result.output_dir / "report.md").read_text()
    )
    record = json.loads((result.output_dir / "candidate.run.json").read_text())[
        "record"
    ]
    facts = record["observed"]["collector"]["facts"]
    assert facts["engine.kv_cache_dtype"] == "fp8"
    assert "CUDA_VISIBLE_DEVICES" not in facts["environment.numerics"]
    assert "GPU-candidate" not in json.dumps(record)


def test_contradicting_collector_evidence_blocks_pass(collectors, tmp_path):
    collectors.kv["candidate"] = (
        "auto"  # server still runs auto despite the declaration
    )
    collectors.start()
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"
    gaps = manifest["records"]["candidate"]["evidence_gaps"]
    assert any("contradicts runtime.declared.kv_cache_dtype" in gap for gap in gaps)


def test_untrusted_signature_blocks_pass(collectors, tmp_path):
    collectors.keys["candidate"] = ROGUE
    collectors.start()
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"
    gaps = manifest["records"]["candidate"]["evidence_gaps"]
    assert any("signature is invalid" in gap for gap in gaps)


def test_unreachable_collector_blocks_pass(collectors, tmp_path, monkeypatch):
    collectors.start()
    monkeypatch.setenv("CAND_COLLECTOR", "http://127.0.0.1:9")
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_facts_changing_mid_run_block_pass(collectors, tmp_path):
    collectors.drivers["candidate"] = ["595.91.07", "600.10.01"]
    collectors.start()
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert any(
        "changed during the run" in gap
        for gap in manifest["records"]["candidate"]["evidence_gaps"]
    )


def test_required_fact_that_is_unavailable_blocks_pass(collectors, tmp_path):
    collectors.start()
    trust = {"require_collector": True, "required_facts": ["image.digest"]}
    manifest = _verify(_recipe(tmp_path, trust=trust), tmp_path / "out").manifest
    assert manifest["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_undeclared_collected_difference_is_not_comparable(collectors, tmp_path):
    collectors.drivers["candidate"] = ["600.10.01"]
    collectors.start()
    manifest = _verify(_recipe(tmp_path), tmp_path / "out").manifest
    assert manifest["verdict"] == "NOT_COMPARABLE"
    assert any(
        "gpu.driver_version" in issue["dimension"]
        for issue in manifest["comparison"]["issues"]
    )


def test_collector_recipe_validation(collectors, tmp_path):
    recipe = _recipe(tmp_path)
    left, right = recipe.study.runs

    def with_runs(new_left, new_right):
        return replace(recipe, study=replace(recipe.study, runs=(new_left, new_right)))

    no_collector = {k: v for k, v in right.runtime.items() if k != "collector"}
    with pytest.raises(ValueError, match="both endpoint runs name a collector"):
        validate_endpoint_recipe(with_runs(left, replace(right, runtime=no_collector)))
    swapped = {
        **right.runtime,
        "collector": {**right.runtime["collector"], "role": "reference"},
    }
    with pytest.raises(ValueError, match="must declare role candidate"):
        validate_endpoint_recipe(with_runs(left, replace(right, runtime=swapped)))
    same_url = {
        **right.runtime,
        "collector": {**right.runtime["collector"], "url_env": "REF_COLLECTOR"},
    }
    with pytest.raises(ValueError, match="different collector URLs"):
        validate_endpoint_recipe(with_runs(left, replace(right, runtime=same_url)))
    other_trust = {
        **right.runtime,
        "collector": {**right.runtime["collector"], "trust": {"max_age_s": 10}},
    }
    with pytest.raises(ValueError, match="same keys_env and trust"):
        validate_endpoint_recipe(with_runs(left, replace(right, runtime=other_trust)))


def test_prepare_endpoint_cli_collector_options(servers, tmp_path):  # noqa: F811
    workload = tmp_path / "w.jsonl"
    workload.write_text("\n".join(json.dumps(row) for row in PROMPTS))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(POLICY.to_data()))
    base = [
        "recipe", "prepare-endpoint", "--model-id", "served-model",
        "--reference-url-env", "REF_URL", "--candidate-url-env", "CAND_URL",
        "--tokenizer", str(tmp_path / "tok"), "--workload", str(workload),
        "--policy", str(policy),
        "--reference-declared", "kv_cache_dtype=auto",
        "--candidate-declared", "kv_cache_dtype=fp8",
    ]  # fmt: skip
    output = tmp_path / "recipe.json"
    code = main(
        base + [
            "--reference-collector-url-env", "REF_COLLECTOR",
            "--candidate-collector-url-env", "CAND_COLLECTOR",
            "--collector-keys-env", "COLLECTOR_KEYS",
            "--reference-deployment", "auto-pool", "--candidate-deployment", "fp8-pool",
            "--required-fact", "engine.kv_cache_dtype", "--output", str(output),
        ],
        stdout=io.StringIO(),
    )  # fmt: skip
    assert code == 0
    runs = json.loads(output.read_text())["study"]["runs"]
    assert runs[1]["runtime"]["collector"]["role"] == "candidate"
    assert runs[1]["runtime"]["collector"]["trust"]["required_facts"] == [
        "engine.kv_cache_dtype"
    ]
    err = io.StringIO()
    partial = main(
        base
        + [
            "--collector-keys-env",
            "COLLECTOR_KEYS",
            "--output",
            str(tmp_path / "x.json"),
        ],
        stdout=io.StringIO(),
        stderr=err,
    )
    assert partial != 0 and "collector options need" in err.getvalue()
