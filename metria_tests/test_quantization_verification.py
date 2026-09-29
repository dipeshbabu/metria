from __future__ import annotations

import copy
import io
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from metria_tests.test_gguf_identity import _model_bytes

from metria import PolicyCriterion, VerificationPolicy
from metria.cli import main
from metria.preparation_quantization import (
    convert_gguf,
    prepare_gguf_quantization_recipe,
)
from metria.runtimes import llamacpp
from metria.runtimes.gguf_identity import inspect_gguf
from metria.verification import verify_recipe
from metria.verification_quantization import (
    CONVERSION_SCHEMA,
    QUANTIZATION_KEY,
)
from metria.verification_schema import GGUF_QUANTIZATION_SCOPE


@pytest.fixture
def quant_case(tmp_path, monkeypatch):
    directory = (tmp_path / "bin").resolve()
    directory.mkdir()
    suffix = ".exe" if os.name == "nt" else ""
    provider = directory / ("llama-completion" + suffix)
    quantizer = directory / ("llama-quantize" + suffix)
    provider.write_bytes(b"qualified CPU capture fixture")
    quantizer.write_bytes(b"qualified quantizer fixture")
    source, candidate = tmp_path / "source.gguf", tmp_path / "candidate.gguf"
    source.write_bytes(_model_bytes())
    candidate.write_bytes(_model_bytes(quantized=True))
    conversion = {
        "schema": CONVERSION_SCHEMA,
        "format": "Q8_0",
        "quantizer_sha256": llamacpp._sha256_file(quantizer),
        "source": {
            "sha256": llamacpp._sha256_file(source),
            "gguf": inspect_gguf(source),
        },
        "candidate": {
            "sha256": llamacpp._sha256_file(candidate),
            "gguf": inspect_gguf(candidate),
        },
    }
    recipe = prepare_gguf_quantization_recipe(
        directory,
        source,
        candidate,
        conversion,
        [
            {
                "id": "task",
                "prompt": "private prompt",
                "checks": [{"id": "present", "kind": "nonempty"}],
            }
        ],
    )
    case = {
        "recipe": recipe,
        "conversion": conversion,
        "source": source.resolve(),
        "candidate": candidate.resolve(),
        "provider": provider,
        "quantizer": quantizer,
        "directory": directory,
        "output": tmp_path / "verification",
        "calls": [],
        "mode": "normal",
    }

    def run(argv, **kwargs):
        model = Path(argv[argv.index("-m") + 1])
        changed = model != source.resolve()
        case["calls"].append(changed)
        if changed and case["mode"] == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        path = Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"])
        tokens = [0, 1, 0 if changed else 1][: int(argv[argv.index("-n") + 1])]
        path.write_text(
            "".join(
                json.dumps({"step": index, "token_id": token}) + "\n"
                for index, token in enumerate(tokens)
            )
        )
        if not (changed and case["mode"] == "missing_capture"):
            capture = {
                "schema": "metria.llamacpp_capture.v1",
                "threads": 2,
                "threads_batch": 1,
                "context": 256,
                "vocab_size": 2,
                "chat_template_applied": False,
            }
            if changed and case["mode"] == "wrong_vocab":
                capture["vocab_size"] = 3
            Path(str(path) + ".runtime.json").write_text(json.dumps(capture))
        if changed and case["mode"] == "changed_artifact":
            model.write_bytes(b"replaced model")
        return subprocess.CompletedProcess(argv, 0, stdout="answer", stderr="")

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


def test_quantization_preserves_tokenizer_controls_and_reports_observed_storage(
    quant_case,
):
    result = verify_recipe(quant_case["recipe"], quant_case["output"])
    data = result.to_data()
    assert result.exit_code == 0 and data["verdict"] == "VERIFIED"
    assert data["scope"] == GGUF_QUANTIZATION_SCOPE
    assert data["change"]["reference"] == {"F32": 1}
    assert data["change"]["candidate"] == {"Q8_0": 1}
    assert data["performance"]["available"]
    assert quant_case["calls"] == [False, True]
    report = (quant_case["output"] / "report.md").read_text()
    assert "Observed tensor storage" in report and "mixed storage" in report
    assert "private prompt" not in report


@pytest.mark.parametrize("threshold,expected", [(0.6, "PASS"), (0.9, "FAIL")])
def test_quantization_policy_uses_measured_behavior(quant_case, threshold, expected):
    recipe = replace(
        quant_case["recipe"],
        policy=VerificationPolicy(
            (
                PolicyCriterion(
                    "behavior.trajectory_agreement", "0.3.4", minimum=threshold
                ),
            )
        ),
    )
    result = verify_recipe(recipe, quant_case["output"])
    assert result.to_data()["verdict"] == expected


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("timeout", "EXECUTION_FAILED"),
        ("missing_capture", "INSUFFICIENT_EVIDENCE"),
        ("wrong_vocab", "INSUFFICIENT_EVIDENCE"),
        ("changed_artifact", "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_unqualified_candidate_never_passes_and_preserves_reference(
    quant_case, mode, expected
):
    quant_case["mode"] = mode
    result = verify_recipe(quant_case["recipe"], quant_case["output"])
    assert result.to_data()["verdict"] == expected
    assert not result.to_data()["performance"]["available"]
    assert (quant_case["output"] / "reference.run.json").is_file()


@pytest.mark.parametrize(
    "case",
    [
        "tokenizer",
        "layout",
        "metadata",
        "format",
        "tool_pin",
        "same_pin",
        "boolean_type",
        "bad_counts",
        "missing_field",
        "extra_field",
    ],
)
def test_conversion_manifest_rejects_invalid_or_uncontrolled_changes(quant_case, case):
    conversion = copy.deepcopy(quant_case["conversion"])
    candidate = conversion["candidate"]
    fields = {
        "tokenizer": "tokenizer_sha256",
        "layout": "tensor_layout_sha256",
        "metadata": "controls_sha256",
    }
    if case in fields:
        candidate["gguf"][fields[case]] = "a" * 64
    elif case == "format":
        conversion["format"] = "Q4_0"
    elif case == "tool_pin":
        conversion["quantizer_sha256"] = "untrusted"
    elif case == "same_pin":
        candidate["sha256"] = conversion["source"]["sha256"]
    elif case == "boolean_type":
        candidate["gguf"]["file_type"] = True
    elif case == "bad_counts":
        candidate["gguf"]["tensor_types"] = {"Q8_0": -1}
    elif case == "missing_field":
        candidate["gguf"].pop("tensor_count")
    elif case == "extra_field":
        conversion["override"] = True
    recipe = replace(
        quant_case["recipe"],
        environment={**quant_case["recipe"].environment, QUANTIZATION_KEY: conversion},
    )
    with pytest.raises(ValueError):
        verify_recipe(recipe, quant_case["output"])
    assert not quant_case["calls"] and not quant_case["output"].exists()


def test_forged_matching_metadata_is_rejected_against_actual_artifacts(quant_case):
    conversion = copy.deepcopy(quant_case["conversion"])
    for role in ("source", "candidate"):
        conversion[role]["gguf"]["tokenizer_sha256"] = "a" * 64
    recipe = replace(
        quant_case["recipe"],
        environment={**quant_case["recipe"].environment, QUANTIZATION_KEY: conversion},
    )
    result = verify_recipe(recipe, quant_case["output"])
    assert result.to_data()["verdict"] == "EXECUTION_FAILED"
    assert not quant_case["calls"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime", {"threads": 3}),
        ("scenario", {"seed": 2}),
        ("model", {"id": "unrelated-model"}),
    ],
)
def test_quantization_cannot_hide_other_requested_changes(quant_case, field, value):
    recipe = quant_case["recipe"]
    left, right = recipe.study.runs
    right = replace(right, **{field: {**getattr(right, field), **value}})
    with pytest.raises(ValueError):
        verify_recipe(
            replace(recipe, study=replace(recipe.study, runs=(left, right))),
            quant_case["output"],
        )
    assert not quant_case["calls"]


def _conversion_runner(monkeypatch, case, mode="normal"):
    import metria.preparation_quantization as preparation

    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if mode != "missing":
            Path(argv[2]).write_bytes(
                _model_bytes(
                    quantized=True,
                    tokens=("a", "changed")
                    if mode == "different_tokenizer"
                    else ("a", "b"),
                )
            )
        if mode == "changed_source":
            case["source"].write_bytes(b"changed source")
        if mode == "changed_tool":
            case["quantizer"].write_bytes(b"changed quantizer")
        if mode == "collision":
            (case["source"].parent / "new.gguf").write_bytes(b"other user's file")
        return SimpleNamespace(
            returncode=1 if mode == "failure" else 0,
            timed_out=mode == "timeout",
            stdout="conversion trace",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
        )

    monkeypatch.setattr(preparation, "run_process", run)
    return calls


def test_conversion_binds_input_tool_output_and_publishes_without_copying_over_files(
    quant_case, tmp_path, monkeypatch
):
    calls = _conversion_runner(monkeypatch, quant_case)
    output, receipt = tmp_path / "new.gguf", tmp_path / "conversion.json"
    result = convert_gguf(
        quant_case["source"],
        quant_case["conversion"]["source"]["sha256"],
        output,
        quant_case["quantizer"],
        receipt,
        quantizer_sha256=quant_case["conversion"]["quantizer_sha256"],
    )
    assert result["candidate"]["sha256"] == llamacpp._sha256_file(output)
    assert json.loads(receipt.read_text())["status"] == "converted"
    assert len(calls) == 1 and calls[0][3:] == ["Q8_0", "2"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        convert_gguf(
            quant_case["source"],
            quant_case["conversion"]["source"]["sha256"],
            output,
            quant_case["quantizer"],
            receipt,
        )
    assert output.read_bytes() == before and len(calls) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "failure",
        "timeout",
        "missing",
        "different_tokenizer",
        "changed_source",
        "changed_tool",
        "collision",
    ],
)
def test_failed_conversion_retains_receipt_without_overwriting_candidate(
    quant_case, tmp_path, monkeypatch, mode
):
    _conversion_runner(monkeypatch, quant_case, mode)
    output, receipt = tmp_path / "new.gguf", tmp_path / "conversion.json"
    with pytest.raises((ValueError, FileExistsError)):
        convert_gguf(
            quant_case["source"],
            quant_case["conversion"]["source"]["sha256"],
            output,
            quant_case["quantizer"],
            receipt,
        )
    assert json.loads(receipt.read_text())["status"] == "failed"
    if mode == "collision":
        assert output.read_bytes() == b"other user's file"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".metria-quantize-*"))


def test_installed_preparation_converts_qualifies_and_preserves_outputs(
    quant_case, tmp_path, monkeypatch
):
    _conversion_runner(monkeypatch, quant_case)
    workload = tmp_path / "workload.jsonl"
    workload.write_text(
        json.dumps(
            {
                "id": "task",
                "prompt": "private prompt",
                "checks": [{"id": "present", "kind": "nonempty"}],
            }
        )
        + "\n"
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            VerificationPolicy(
                (PolicyCriterion("quality.candidate_pass_rate", "1", minimum=1.0),)
            ).to_data()
        )
    )
    args = [
        "recipe",
        "prepare-gguf-quantization",
        "--bin-dir",
        str(quant_case["directory"]),
        "--model",
        str(quant_case["source"]),
        "--model-sha256",
        quant_case["conversion"]["source"]["sha256"],
        "--candidate-model",
        str(tmp_path / "new.gguf"),
        "--workload",
        str(workload),
        "--policy",
        str(policy),
        "--output",
        str(tmp_path / "study.json"),
    ]
    out, err = io.StringIO(), io.StringIO()
    assert main(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "Q8_0" in out.getvalue()
    assert (tmp_path / "study.conversion.json").is_file()
    assert (tmp_path / "study.reference.qualification.run.json").is_file()
    assert (tmp_path / "study.candidate.qualification.run.json").is_file()
    assert main(args, stdout=out, stderr=err) == 2
    assert quant_case["calls"] == [False, True]


@pytest.mark.parametrize(
    "options",
    [
        {"threads": 0},
        {"threads": True},
        {"timeout": float("inf")},
        {"timeout": 0},
        {"quantizer_sha256": "a" * 64},
    ],
)
def test_invalid_conversion_options_fail_before_native_execution(
    quant_case, tmp_path, monkeypatch, options
):
    calls = _conversion_runner(monkeypatch, quant_case)
    with pytest.raises(ValueError):
        convert_gguf(
            quant_case["source"],
            quant_case["conversion"]["source"]["sha256"],
            tmp_path / "new.gguf",
            quant_case["quantizer"],
            tmp_path / "receipt.json",
            **options,
        )
    assert not calls


def test_source_pin_and_output_aliases_are_rejected_before_conversion(
    quant_case, tmp_path, monkeypatch
):
    calls = _conversion_runner(monkeypatch, quant_case)
    with pytest.raises(ValueError, match="trusted SHA256"):
        convert_gguf(
            quant_case["source"],
            "a" * 64,
            tmp_path / "new.gguf",
            quant_case["quantizer"],
            tmp_path / "receipt.json",
        )
    with pytest.raises(ValueError, match="separate paths"):
        convert_gguf(
            quant_case["source"],
            quant_case["conversion"]["source"]["sha256"],
            tmp_path / "new.gguf",
            quant_case["quantizer"],
            tmp_path / "new.gguf",
        )
    assert not calls


def test_requantization_is_not_mislabeled_as_original_weight_conversion(
    quant_case, tmp_path, monkeypatch
):
    calls = _conversion_runner(monkeypatch, quant_case)
    with pytest.raises(ValueError, match="already quantized"):
        convert_gguf(
            quant_case["candidate"],
            quant_case["conversion"]["candidate"]["sha256"],
            tmp_path / "new.gguf",
            quant_case["quantizer"],
            tmp_path / "receipt.json",
        )
    assert not calls


@pytest.mark.parametrize(
    "case",
    [
        "one_run",
        "broad_variation",
        "unknown_analysis",
        "missing_provider",
        "missing_workload",
        "wrong_model_pin",
        "relative_provider",
        "relative_model",
        "same_model_path",
    ],
)
def test_recipe_structure_cannot_bypass_quantization_scope(quant_case, case):
    recipe = quant_case["recipe"]
    left, right = recipe.study.runs
    if case == "one_run":
        recipe = replace(recipe, study=replace(recipe.study, runs=(left,)))
    elif case == "broad_variation":
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                comparison=replace(recipe.study.comparison, vary=frozenset({"model"})),
            ),
        )
    elif case == "unknown_analysis":
        recipe = replace(
            recipe,
            study=replace(
                recipe.study,
                comparison=replace(recipe.study.comparison, analyses=("unknown",)),
            ),
        )
    elif case == "missing_provider":
        recipe = replace(
            recipe, environment={QUANTIZATION_KEY: quant_case["conversion"]}
        )
    elif case == "missing_workload":
        recipe = replace(recipe, measurement_configs={})
    else:
        if case == "wrong_model_pin":
            right = replace(right, model={**right.model, "sha256": "a" * 64})
        elif case == "relative_provider":
            right = replace(right, runtime={**right.runtime, "bin_dir": "relative"})
        elif case == "relative_model":
            right = replace(right, model={**right.model, "path": "relative.gguf"})
        elif case == "same_model_path":
            right = replace(right, model={**right.model, "path": left.model["path"]})
        recipe = replace(recipe, study=replace(recipe.study, runs=(left, right)))
    with pytest.raises(ValueError):
        verify_recipe(recipe, quant_case["output"])
    assert not quant_case["calls"] and not quant_case["output"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unqualified"),
        ("architecture", None),
        ("vocab_size", True),
        ("tensor_types", {"Q4_0": 1}),
        ("file_type", 0),
        ("quantization_version", 1),
    ],
)
def test_unqualified_candidate_inspection_is_not_trusted(quant_case, field, value):
    conversion = copy.deepcopy(quant_case["conversion"])
    conversion["candidate"]["gguf"][field] = value
    recipe = replace(
        quant_case["recipe"],
        environment={**quant_case["recipe"].environment, QUANTIZATION_KEY: conversion},
    )
    with pytest.raises(ValueError):
        verify_recipe(recipe, quant_case["output"])
    assert not quant_case["calls"]
