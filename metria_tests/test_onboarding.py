import json
from io import StringIO

import pytest
from metria_tests.test_vllm_verify import prefix_case as _prefix_case_fixture

from metria.cli import main

prefix_case = _prefix_case_fixture


@pytest.mark.parametrize(
    "case,code,verdict",
    [("pass", 0, "PASS"), ("fail", 1, "FAIL"), ("not-comparable", 3, "NOT_COMPARABLE")],
)
def test_explicit_demo_keeps_synthetic_scope_and_real_exit_contract(
    tmp_path, case, code, verdict
):
    output = tmp_path / case
    stdout, stderr = StringIO(), StringIO()
    assert (
        main(
            ["demo", "--case", case, "--output", str(output)],
            stdout=stdout,
            stderr=stderr,
        )
        == code
    )
    data = json.loads((output / "verification.json").read_text())
    assert data["fixture_only"] is True and data["scope"] == "synthetic_fixture.v1"
    assert data["verdict"] == verdict
    assert "synthetic fixture" in stdout.getvalue()


def test_demo_preserves_existing_output(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    stderr = StringIO()
    assert main(["demo", "--output", str(tmp_path)], stderr=stderr) == 2
    assert marker.read_text() == "keep"
    assert "already exists" in stderr.getvalue()


def test_preparation_requires_explicit_inputs_or_named_example(tmp_path):
    stderr = StringIO()
    code = main(
        [
            "recipe",
            "prepare-vllm",
            "--model",
            str(tmp_path),
            "--output",
            str(tmp_path / "study.json"),
        ],
        stderr=stderr,
    )
    assert code == 2 and "--descriptor" in stderr.getvalue()
    assert not (tmp_path / "study.json").exists()


def test_example_manifest_is_available_without_a_checkout(tmp_path, monkeypatch):
    from metria import onboarding

    monkeypatch.setattr(onboarding, "_runtime_host_supported", lambda: True)
    from importlib import resources

    descriptor = json.loads(
        resources.files("metria").joinpath("data/smollm2-135m.json").read_text()
    )
    assert descriptor["revision"] == "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    stderr = StringIO()
    code = main(
        [
            "recipe",
            "prepare-vllm",
            "--example",
            "--model",
            str(tmp_path),
            "--output",
            str(tmp_path / "study.json"),
        ],
        stderr=stderr,
    )
    assert code == 2 and "inventory" in stderr.getvalue()
    assert not (tmp_path / "study.json").exists()


def test_preparation_reports_missing_runtime_without_launching_a_model(
    tmp_path, monkeypatch
):
    import importlib.metadata

    from metria import onboarding

    monkeypatch.setattr(onboarding, "_runtime_host_supported", lambda: True)

    def unavailable(*args, **kwargs):
        raise importlib.metadata.PackageNotFoundError("vllm")

    monkeypatch.setattr(onboarding, "prepare_vllm_prefix_recipe", unavailable)
    stderr = StringIO()
    code = main(
        [
            "recipe",
            "prepare-vllm",
            "--example",
            "--model",
            str(tmp_path),
            "--output",
            str(tmp_path / "study.json"),
        ],
        stderr=stderr,
    )
    assert code == 2 and "qualified vLLM environment" in stderr.getvalue()
    assert not (tmp_path / "study.json").exists()


def test_preparation_does_not_overwrite_a_recipe(tmp_path):
    output = tmp_path / "study.json"
    output.write_text("keep")
    assert (
        main(
            [
                "recipe",
                "prepare-vllm",
                "--example",
                "--model",
                str(tmp_path),
                "--output",
                str(output),
            ],
            stderr=StringIO(),
        )
        == 2
    )
    assert output.read_text() == "keep"


def test_invalid_jsonl_rejected_before_runtime_inspection(tmp_path, monkeypatch):
    from metria import onboarding

    workload = tmp_path / "workload.jsonl"
    workload.write_text('{"id":"first","id":"duplicate","prompt":"private"}\n')
    monkeypatch.setattr(
        onboarding,
        "prepare_vllm_prefix_recipe",
        lambda *a, **kw: pytest.fail("runtime preparation must not start"),
    )
    stderr = StringIO()
    assert (
        main(
            [
                "recipe",
                "prepare-vllm",
                "--example",
                "--model",
                str(tmp_path),
                "--workload",
                str(workload),
                "--output",
                str(tmp_path / "study.json"),
            ],
            stderr=stderr,
        )
        == 2
    )
    assert "private" not in stderr.getvalue()


def test_prepare_command_writes_a_valid_bound_recipe_with_spaced_paths(
    prefix_case, tmp_path, monkeypatch
):
    from metria import onboarding, preparation

    monkeypatch.setattr(onboarding, "_runtime_host_supported", lambda: True)
    from metria.recipes import load_study_recipe

    monkeypatch.setattr(
        preparation,
        "installed_runtime_identity",
        lambda: {"version": "0.30.0+cpu", "sha256": "a" * 64},
    )
    descriptor = tmp_path / "trusted descriptor.json"
    descriptor.write_text(
        json.dumps({"files": dict(prefix_case["recipe"].study.runs[0].model["files"])})
    )
    workload = tmp_path / "my workload.jsonl"
    workload.write_text(
        '{"id":"one","prompt":"shared prefix one"}\n{"id":"two","prompt":"shared prefix two"}\n'
    )
    output = tmp_path / "my study.json"
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "metria.verification_policy.v1",
                "criteria": [
                    {"target": "quality.candidate_pass_rate", "version": "1", "min": 0}
                ],
            }
        )
    )
    stdout, stderr = StringIO(), StringIO()
    code = main(
        [
            "recipe",
            "prepare-vllm",
            "--model",
            prefix_case["recipe"].study.runs[0].model["path"],
            "--descriptor",
            str(descriptor),
            "--workload",
            str(workload),
            "--output",
            str(output),
            "--policy",
            str(policy),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0, stderr.getvalue()
    recipe = load_study_recipe(output)
    assert recipe.environment["vllm_distribution_sha256"] == "a" * 64
    assert recipe.study.runs[0].runtime["enable_prefix_caching"] is False
    assert recipe.study.runs[1].runtime["enable_prefix_caching"] is True
    assert not prefix_case["engines"]
    assert recipe.policy.criteria[0].target == "quality.candidate_pass_rate"
    assert "Next: metria verify" in stdout.getvalue()


def test_non_object_descriptor_has_an_actionable_error(tmp_path):
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("[]")
    stderr = StringIO()
    code = main(
        [
            "recipe",
            "prepare-vllm",
            "--example",
            "--model",
            str(tmp_path),
            "--descriptor",
            str(descriptor),
            "--output",
            str(tmp_path / "study.json"),
        ],
        stderr=stderr,
    )
    assert code == 2 and "JSON object" in stderr.getvalue()


def test_unsupported_host_fails_before_runtime_inspection(tmp_path, monkeypatch):
    from metria import onboarding

    monkeypatch.setattr(onboarding, "_runtime_host_supported", lambda: False)
    monkeypatch.setattr(
        onboarding,
        "prepare_vllm_prefix_recipe",
        lambda *a, **kw: pytest.fail(
            "unsupported host must not inspect or launch the runtime"
        ),
    )
    stderr = StringIO()
    assert (
        main(
            [
                "recipe",
                "prepare-vllm",
                "--example",
                "--model",
                str(tmp_path),
                "--output",
                str(tmp_path / "study.json"),
            ],
            stderr=stderr,
        )
        == 2
    )
    assert "Linux (including WSL)" in stderr.getvalue()


def test_unlabelled_demo_artifact_is_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from metria import onboarding

    output = tmp_path / "invalid demo"

    def fake_process(*args, **kwargs):
        output.mkdir()
        (output / "verification.json").write_text(
            json.dumps({"scope": "real", "fixture_only": False, "exit_code": 0})
        )
        return SimpleNamespace(stdout="", stderr="", timed_out=False, returncode=0)

    monkeypatch.setattr(onboarding, "run_process", fake_process)
    stderr = StringIO()
    assert main(["demo", "--output", str(output)], stderr=stderr) == 5
    assert "labelled result" in stderr.getvalue()
