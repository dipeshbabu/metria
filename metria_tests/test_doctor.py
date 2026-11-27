"""``metria doctor`` diagnostics and ``metria init`` starter files."""

from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from metria import doctor
from metria.cli import main
from metria.policies import policy_from_data
from metria.preparation_endpoint import prepare_endpoint_recipe


def _cli(*argv: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue() + err.getvalue()


def _versions(mapping):
    return lambda name: mapping.get(name)


def test_packages_report_qualified_profiles_and_reject_unqualified_vllm():
    good = doctor.check_packages(_versions({"vllm": "0.30.0+cu129", "torch": "2.9"}))
    qualified = next(c for c in good if c.name == "vllm.qualified")
    assert qualified.status == "ok"
    assert "FP8 KV cache" in qualified.detail and "parallelism" in qualified.detail
    cpu = doctor.check_packages(_versions({"vllm": "0.30.0+cpu"}))
    detail = next(c for c in cpu if c.name == "vllm.qualified").detail
    assert "CPU runtime upgrade" in detail and "FP8" not in detail
    bad = doctor.check_packages(_versions({"vllm": "0.31.0"}))
    assert next(c for c in bad if c.name == "vllm.qualified").status == "fail"
    missing = doctor.check_packages(_versions({}))
    assert all(c.status == "warn" for c in missing)
    assert not any(c.name == "vllm.qualified" for c in missing)


def test_gpu_checks_cover_fp8_parallelism_and_absence():
    two = doctor.check_gpus(lambda _: "NVIDIA L4, 8.9\nNVIDIA L4, 8.9\n")
    assert [c.status for c in two] == ["ok", "ok"]
    one = doctor.check_gpus(lambda _: "NVIDIA A100, 8.0\n")
    statuses = {c.name: c.status for c in one}
    assert statuses == {"gpu": "ok", "gpu.fp8": "warn", "gpu.parallel": "warn"}
    assert doctor.check_gpus(lambda _: "\n")[0].status == "warn"

    def absent(_):
        raise FileNotFoundError("nvidia-smi")

    assert doctor.check_gpus(absent)[0].status == "warn"

    def broken(_):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    assert doctor.check_gpus(broken)[0].status == "warn"


def test_model_dir_checks(tmp_path):
    assert doctor.check_model_dir(tmp_path / "missing")[0].status == "fail"
    statuses = {c.name: c.status for c in doctor.check_model_dir(tmp_path)}
    assert statuses == {"model.config": "warn", "model.tokenizer": "fail"}
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    assert all(c.status == "ok" for c in doctor.check_model_dir(tmp_path))


def test_endpoint_env_checks_never_print_values(monkeypatch):
    monkeypatch.setenv("REF_URL", "https://secret-host.example/v1")
    monkeypatch.setenv("BAD_URL", "ftp://example")
    monkeypatch.delenv("UNSET_URL", raising=False)
    checks = doctor.check_endpoint_env(["REF_URL", "BAD_URL", "UNSET_URL"])
    assert [c.status for c in checks] == ["ok", "fail", "fail"]
    assert "secret-host" not in json.dumps([c.as_mapping() for c in checks])


def test_policy_check(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(doctor._POLICY_TEMPLATE))
    assert doctor.check_policy(good)[0].status == "ok"
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "nope"}')
    assert doctor.check_policy(bad)[0].status == "fail"
    assert doctor.check_policy(tmp_path / "missing.json")[0].status == "fail"


def test_python_version_bounds(monkeypatch):
    monkeypatch.setattr(doctor.sys, "version_info", (3, 9, 0))
    assert doctor.check_python()[1].status == "fail"
    monkeypatch.setattr(doctor.sys, "version_info", (3, 10, 4))
    assert doctor.check_python()[1].status == "warn"


def test_doctor_cli_json_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("REF_URL", "http://127.0.0.1:8000")
    code, output = _cli("doctor", "--no-gpus", "--json", "--endpoint-env", "REF_URL")
    report = json.loads(output)
    assert report["schema"] == "metria.doctor.v1"
    assert report["ok"] is (code == 0)
    code, output = _cli("doctor", "--no-gpus", "--model", str(tmp_path / "none"))
    assert code == 1
    assert "FAIL" in output and "fix:" in output and "blocking problem" in output


def test_doctor_text_success(monkeypatch):
    monkeypatch.setattr(doctor, "check_packages", lambda: [])
    monkeypatch.setattr(doctor, "check_gpus", lambda: [doctor.Check("gpu", "ok", "x")])
    args = SimpleNamespace(
        no_gpus=False, model=None, endpoint_env=[], policy=None, json_output=False
    )
    out = io.StringIO()
    assert doctor.doctor_command(args, out) == 0
    assert "No blocking problems" in out.getvalue()


def test_init_writes_reviewable_files_that_prepare_a_recipe(tmp_path):
    model = tmp_path / "tok"
    model.mkdir()
    (model / "tokenizer.json").write_text('{"model": {}}')
    (model / "config.json").write_text("{}")
    (model / "README.md").write_text("not pinned")
    target = tmp_path / "starter"
    code, output = _cli("init", "--output-dir", str(target), "--model", str(model))
    assert code == 0, output
    names = sorted(p.name for p in target.iterdir())
    assert names == [
        "NEXT_STEPS.md",
        "descriptor.proposed.json",
        "policy.json",
        "workload.jsonl",
    ]
    pins = json.loads((target / "descriptor.proposed.json").read_text())
    assert set(pins["files"]) == {"config.json", "tokenizer.json"}
    assert pins["model_id"] == "tok"
    assert "--approve-pins" in (target / "NEXT_STEPS.md").read_text()
    prompts = [
        json.loads(line)
        for line in (target / "workload.jsonl").read_text().splitlines()
    ]
    recipe = prepare_endpoint_recipe(
        model_id="served",
        tokenizer_dir=model,
        prompts=prompts,
        reference_url_env="REF",
        candidate_url_env="CAND",
        reference_declared={"kv_cache_dtype": "auto"},
        candidate_declared={"kv_cache_dtype": "fp8"},
        policy=policy_from_data(json.loads((target / "policy.json").read_text())),
    )
    assert recipe.policy is not None
    code, output = _cli("init", "--output-dir", str(target))
    assert code != 0 and "already exists" in output


def test_init_approved_pins_and_bad_model(tmp_path):
    model = tmp_path / "tok"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    target = tmp_path / "approved"
    assert (
        _cli(
            "init", "--output-dir", str(target), "--model", str(model), "--approve-pins"
        )[0]
        == 0
    )
    assert (target / "descriptor.json").is_file()
    assert "--approve-pins" not in (target / "NEXT_STEPS.md").read_text()
    code, output = _cli(
        "init", "--output-dir", str(tmp_path / "x"), "--model", str(tmp_path / "no")
    )
    assert code != 0 and not (tmp_path / "x").exists()


def test_commands_are_listed_in_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    output = capsys.readouterr().out
    assert "doctor" in output and "init" in output
