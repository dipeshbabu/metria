from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from metria import PolicyCriterion, VerificationPolicy, load_study_recipe
from metria.cli import main
from metria.preparation_cpu import prepare_llamacpp_build_recipe
from metria.runtimes import llamacpp
from metria.runtimes.llamacpp_qualified import PROVIDERS_KEY
from metria.verification import verify_recipe
from metria.verification_schema import LLAMACPP_BUILD_SCOPE
from metria.verification_trials import (
    VerificationTrialPolicy,
    execute_verification_trials,
)


@pytest.fixture
def build_case(tmp_path, monkeypatch):
    binaries = []
    for role in ("reference", "candidate"):
        directory = tmp_path / role
        directory.mkdir()
        binary = directory / (
            "llama-completion.exe" if os.name == "nt" else "llama-completion"
        )
        binary.write_bytes(f"qualified {role} fixture".encode())
        binaries.append(binary.resolve())
    model = tmp_path / "model.gguf"
    model.write_bytes(b"immutable shared GGUF fixture")
    model_pin = llamacpp._sha256_file(model)
    prompts = [
        {
            "id": "task",
            "prompt": "private-workload-text",
            "checks": [{"id": "present", "kind": "nonempty"}],
        }
    ]
    recipe = prepare_llamacpp_build_recipe(
        binaries[0].parent,
        binaries[1].parent,
        model,
        model_pin,
        prompts,
        context=256,
        max_tokens=3,
    )
    case = {
        "recipe": recipe,
        "binaries": binaries,
        "mode": "normal",
        "calls": [],
        "output": tmp_path / "result",
    }

    def run(argv, **kwargs):
        candidate = Path(argv[0]).parent == binaries[1].parent
        case["calls"].append(candidate)
        if candidate and case["mode"] == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        path = Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"])
        tokens = [11, 12, 14 if candidate else 13][: int(argv[argv.index("-n") + 1])]
        path.write_text(
            "".join(
                json.dumps({"step": i, "token_id": t}) + "\n"
                for i, t in enumerate(tokens)
            )
        )
        if not (candidate and case["mode"] == "missing_capture"):
            facts = {
                "schema": "metria.llamacpp_capture.v1",
                "threads": 2,
                "threads_batch": 1,
                "context": 256,
                "vocab_size": 512,
                "chat_template_applied": False,
            }
            if candidate and case["mode"] == "wrong_threads":
                facts["threads"] = 8
            if candidate and case["mode"] == "different_vocab":
                facts["vocab_size"] = 1024
            Path(str(path) + ".runtime.json").write_text(json.dumps(facts))
        if candidate and case["mode"] == "replace_provider":
            binaries[1].write_bytes(b"unqualified replacement")
        if candidate and case["mode"] == "replace_model":
            model.write_bytes(b"unqualified replacement model")
        return subprocess.CompletedProcess(
            argv, 0, stdout="generated-answer", stderr=""
        )

    monkeypatch.setattr(
        llamacpp,
        "subprocess",
        SimpleNamespace(
            run=run,
            DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    return case


def test_build_comparison_retains_distinct_pins_and_matching_controls(build_case):
    result = verify_recipe(build_case["recipe"], build_case["output"])
    data = result.to_data()
    assert result.exit_code == 0
    assert data["scope"] == LLAMACPP_BUILD_SCOPE
    assert data["verdict"] == "VERIFIED"
    assert data["change"]["reference"] != data["change"]["candidate"]
    assert data["performance"]["available"] is True
    assert build_case["calls"] == [False, True]
    assert (build_case["output"] / "reference.run.json").is_file()
    assert (build_case["output"] / "candidate.run.json").is_file()
    report = (build_case["output"] / "report.md").read_text()
    assert "CPU build comparison" in report
    assert "Capture provider SHA256" in report
    assert "private-workload-text" not in report


def test_repeated_build_comparison_binds_both_providers_to_baseline(build_case):
    result = execute_verification_trials(
        build_case["recipe"], build_case["output"], policy=VerificationTrialPolicy(0, 2)
    )
    assert result["exit_code"] == 0
    assert result["baseline_key"] is not None
    assert len(result["pairs"]) == 2
    assert build_case["calls"] == [False, True, False, True]


@pytest.mark.parametrize("minimum,expected", [(0.6, "PASS"), (0.9, "FAIL")])
def test_build_route_applies_policy_only_after_valid_comparison(
    build_case, minimum, expected
):
    recipe = replace(
        build_case["recipe"],
        policy=VerificationPolicy(
            (
                PolicyCriterion(
                    "behavior.trajectory_agreement", "0.3.4", minimum=minimum
                ),
            )
        ),
    )
    result = verify_recipe(recipe, build_case["output"])
    assert result.to_data()["verdict"] == expected


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("missing_capture", "INSUFFICIENT_EVIDENCE"),
        ("wrong_threads", "INSUFFICIENT_EVIDENCE"),
        ("replace_provider", "INSUFFICIENT_EVIDENCE"),
        ("replace_model", "INSUFFICIENT_EVIDENCE"),
        ("different_vocab", "NOT_COMPARABLE"),
        ("timeout", "EXECUTION_FAILED"),
    ],
)
def test_bad_candidate_never_passes_and_reference_evidence_survives(
    build_case, mode, expected
):
    build_case["mode"] = mode
    result = verify_recipe(build_case["recipe"], build_case["output"])
    assert result.to_data()["verdict"] == expected
    assert (build_case["output"] / "reference.run.json").is_file()
    assert not result.to_data()["performance"]["available"]


def test_provider_changed_between_resolution_and_launch_is_rejected(
    build_case, monkeypatch
):
    original = llamacpp.LlamaCppAdapter.resolve

    def replace_after_resolution(adapter, spec, environment):
        resolved = original(adapter, spec, environment)
        if spec.runtime["bin_dir"] == str(build_case["binaries"][1].parent):
            build_case["binaries"][1].write_bytes(b"changed after pin check")
        return resolved

    monkeypatch.setattr(llamacpp.LlamaCppAdapter, "resolve", replace_after_resolution)
    result = verify_recipe(build_case["recipe"], build_case["output"])
    assert result.to_data()["verdict"] == "EXECUTION_FAILED"
    assert build_case["calls"] == [False]


@pytest.mark.parametrize(
    "case",
    [
        "missing_pin",
        "same_digest",
        "invalid_digest",
        "relative_directory",
        "extra_environment",
        "broad_variation",
        "extra_analysis",
        "no_workload",
        "one_run",
        "different_threads",
        "different_model",
        "same_build",
        "unmapped_build",
    ],
)
def test_build_recipe_rejects_unsupported_or_ambiguous_changes_before_launch(
    build_case, case
):
    recipe = build_case["recipe"]
    left, right = recipe.study.runs
    providers = dict(recipe.environment[PROVIDERS_KEY])
    if case == "missing_pin":
        providers.pop(right.runtime["bin_dir"])
    elif case == "same_digest":
        providers[right.runtime["bin_dir"]] = providers[left.runtime["bin_dir"]]
    elif case == "invalid_digest":
        providers[right.runtime["bin_dir"]] = "invalid"
    elif case == "relative_directory":
        providers["relative"] = providers.pop(right.runtime["bin_dir"])
    elif case == "extra_environment":
        recipe = replace(recipe, environment={**recipe.environment, "undeclared": True})
    elif case == "broad_variation":
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                comparison=replace(
                    recipe.study.comparison, vary=frozenset({"runtime"})
                ),
            ),
        )
    elif case == "extra_analysis":
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                comparison=replace(recipe.study.comparison, analyses=("unknown",)),
            ),
        )
    elif case == "no_workload":
        recipe = replace(recipe, measurement_configs={})
    elif case == "one_run":
        recipe = replace(recipe, study=replace(recipe.study, runs=(left,)))
    else:
        if case == "different_threads":
            right = replace(right, runtime={**right.runtime, "threads": 3})
        elif case == "different_model":
            right = replace(right, model={**right.model, "id": "another-model"})
        elif case == "same_build":
            right = left
        elif case == "unmapped_build":
            right = replace(right, runtime={**right.runtime, "bin_dir": "/unmapped"})
        recipe = replace(recipe, study=replace(recipe.study, runs=(left, right)))
    if case in {"missing_pin", "same_digest", "invalid_digest", "relative_directory"}:
        recipe = replace(recipe, environment={PROVIDERS_KEY: providers})
    with pytest.raises(ValueError):
        verify_recipe(recipe, build_case["output"])
    assert not build_case["calls"]
    assert not build_case["output"].exists()


def _prepare_args(case, tmp_path):
    recipe = case["recipe"]
    workload = tmp_path / "prompts.jsonl"
    workload.write_text(
        json.dumps(
            {
                "id": "task",
                "prompt": "private-workload-text",
                "checks": [{"id": "present", "kind": "nonempty"}],
            }
        )
        + "\n"
    )
    return [
        "recipe",
        "prepare-llamacpp-build",
        "--reference-bin-dir",
        str(case["binaries"][0].parent),
        "--candidate-bin-dir",
        str(case["binaries"][1].parent),
        "--model",
        recipe.study.runs[0].model["path"],
        "--model-sha256",
        recipe.study.runs[0].model["sha256"],
        "--workload",
        str(workload),
        "--output",
        str(tmp_path / "study.json"),
    ]


def test_installed_preparation_qualifies_both_providers_and_preserves_files(
    build_case, tmp_path
):
    args = _prepare_args(build_case, tmp_path)
    out, err = io.StringIO(), io.StringIO()
    assert main(args, stdout=out, stderr=err) == 0
    recipe = load_study_recipe(tmp_path / "study.json")
    assert len(recipe.environment[PROVIDERS_KEY]) == 2
    assert build_case["calls"] == [False, True]
    for role in ("reference", "candidate"):
        assert (tmp_path / f"study.{role}.qualification.run.json").is_file()
    original = (tmp_path / "study.json").read_bytes()
    assert main(args, stdout=out, stderr=err) == 2
    assert (tmp_path / "study.json").read_bytes() == original
    assert build_case["calls"] == [False, True]


def test_failed_preparation_retains_both_attempts_without_writing_recipe(
    build_case, tmp_path
):
    build_case["mode"] = "missing_capture"
    err = io.StringIO()
    assert (
        main(_prepare_args(build_case, tmp_path), stdout=io.StringIO(), stderr=err) == 2
    )
    assert "candidate provider qualification failed" in err.getvalue()
    assert not (tmp_path / "study.json").exists()
    assert (tmp_path / "study.reference.qualification.run.json").is_file()
    assert (tmp_path / "study.candidate.qualification.run.json").is_file()


def test_preparation_reports_missing_completion_binary(build_case):
    build_case["binaries"][1].unlink()
    recipe = build_case["recipe"]
    with pytest.raises(ValueError, match="patched llama-completion"):
        prepare_llamacpp_build_recipe(
            build_case["binaries"][0].parent,
            build_case["binaries"][1].parent,
            Path(recipe.study.runs[0].model["path"]),
            recipe.study.runs[0].model["sha256"],
            [{"id": "task", "prompt": "test"}],
        )


@pytest.mark.parametrize(
    "field,values",
    [
        ("runtime", {"unmanaged_flag": True}),
        ("runtime", {"n_gpu_layers": 1}),
        ("runtime", {"extra_args": ["--unmanaged"]}),
        ("runtime", {"threads": None}),
        ("model", {"unbound_tokenizer": "different"}),
        ("model", {"sha256": None}),
        ("scenario", {"system": "unqualified system prompt"}),
        ("scenario", {"top_k": 10}),
        ("scenario", {"temperature": 1.0}),
        ("scenario", {"max_tokens": 0}),
    ],
)
def test_build_profile_preserves_shared_cpu_input_restrictions(
    build_case, field, values
):
    recipe = build_case["recipe"]
    runs = tuple(
        replace(run, **{field: {**getattr(run, field), **values}})
        for run in recipe.study.runs
    )
    recipe = replace(recipe, study=replace(recipe.study, runs=runs))
    with pytest.raises(ValueError):
        verify_recipe(recipe, build_case["output"])
    assert not build_case["calls"]
    assert not build_case["output"].exists()


@pytest.mark.parametrize(
    "expected,verdict", [("generated-answer", "PASS"), ("different-answer", "FAIL")]
)
def test_build_task_checks_evaluate_real_outputs_under_quality_policy(
    build_case, expected, verdict
):
    from metria.measurements import TokenTrajectoryProtocol

    recipe = build_case["recipe"]
    config = {
        "prompts": [
            {
                "id": "task",
                "prompt": "private-workload-text",
                "checks": [
                    {"id": "answer", "kind": "exact_text", "expected": expected}
                ],
            }
        ]
    }
    recipe = replace(
        recipe,
        measurement_configs={TokenTrajectoryProtocol.name: config},
        policy=VerificationPolicy(
            (PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),)
        ),
    )
    result = verify_recipe(recipe, build_case["output"])
    assert result.to_data()["verdict"] == verdict
    report = (build_case["output"] / "report.md").read_text()
    assert "Task checks:" in report
    assert "generated-answer" not in report
    assert "different-answer" not in report


def test_missing_declared_task_check_evidence_cannot_be_verified(
    build_case, monkeypatch
):
    from metria.measurements.checked_trajectory import CheckedTrajectoryProtocol
    from metria.protocols import MeasurementResult

    original = CheckedTrajectoryProtocol.execute

    def omit_quality(protocol, session, scenario, config):
        result = original(protocol, session, scenario, config)
        return MeasurementResult(
            metrics=result.metrics,
            evidence={
                key: value
                for key, value in result.evidence.items()
                if key != "task_checks"
            },
        )

    monkeypatch.setattr(CheckedTrajectoryProtocol, "execute", omit_quality)
    result = verify_recipe(build_case["recipe"], build_case["output"])
    assert result.to_data()["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_unconfigured_build_checks_do_not_invent_quality(build_case):
    from metria.measurements import TokenTrajectoryProtocol

    recipe = replace(
        build_case["recipe"],
        measurement_configs={
            TokenTrajectoryProtocol.name: {
                "prompts": [{"id": "task", "prompt": "private-workload-text"}]
            }
        },
    )
    result = verify_recipe(recipe, build_case["output"])
    assert result.to_data()["verdict"] == "VERIFIED"
    report = (build_case["output"] / "report.md").read_text()
    assert "Task quality unavailable" in report


def test_retained_build_evidence_preserves_original_file_and_record_digests():
    from metria.records import load_run_record, run_record_digest

    root = Path(__file__).parents[1] / "artifacts/qualification/llamacpp-builds"
    index = json.loads((root / "files-sha256.json").read_text())
    for relative, expected in index["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    result = json.loads((root / "verification/verification.json").read_text())
    assert result["verdict"] == "PASS"
    for role in ("reference", "candidate"):
        record = load_run_record(root / "verification" / f"{role}.run.json")
        assert run_record_digest(record) == result["records"][role]["record_digest"]
