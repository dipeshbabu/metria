"""Controlled verification of materialized GGUF Q8_0 weight quantization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from .measurements.checked_trajectory import CheckedTrajectoryProtocol
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord
from .recipes import StudyRecipe
from .runtimes.llamacpp_quantized import QuantizedLlamaCppAdapter
from .verification_cpu import validate_cpu_run
from .verification_evidence import task_check_gaps, task_check_identity
from .verification_route import VerificationRoute
from .verification_schema import GGUF_QUANTIZATION_SCOPE

QUANTIZATION_KEY = "gguf_quantization"
CAPTURE_KEY = "llama_cpp_token_ids_capture_sha256"
CONVERSION_SCHEMA = "metria.gguf_quantization.v1"
QUANTIZATION_VARIATIONS = frozenset(
    {
        "model.path",
        "model.sha256",
        *(
            f"{root}.{field}"
            for root in ("resolved.model", "observed.model")
            for field in (
                "path",
                "sha256",
                "size_bytes",
                "mtime_ns",
                "requested_sha256",
            )
        ),
        *(
            f"observed.identity.model.{field}"
            for field in ("path", "sha256", "size_bytes", "mtime_ns")
        ),
        *(
            f"{root}.gguf.{field}"
            for root in ("resolved", "observed")
            for field in ("file_type", "quantization_version", "tensor_types")
        ),
    }
)


def _pin(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            "quantization artifact and tool pins must be lowercase SHA256 digests"
        )
    return value


def _inspection(value: Any) -> Mapping[str, Any]:
    fields = {
        "schema",
        "format",
        "architecture",
        "file_type",
        "quantization_version",
        "vocab_size",
        "tensor_count",
        "tensor_types",
        "tokenizer_sha256",
        "controls_sha256",
        "tensor_layout_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("conversion requires a complete GGUF inspection")
    if (
        value["schema"] != "metria.gguf_identity.v1"
        or value["format"] != "GGUFv3-little-endian"
    ):
        raise ValueError(
            "conversion manifest uses an unsupported GGUF inspection method"
        )
    for name in ("file_type", "quantization_version"):
        if value[name] is not None and type(value[name]) is not int:
            raise ValueError(
                "GGUF type and quantization version must be integers or unavailable"
            )
    if not isinstance(value["architecture"], str) or not value["architecture"]:
        raise ValueError("GGUF architecture identity is required")
    for name in ("vocab_size", "tensor_count"):
        if type(value[name]) is not int or value[name] <= 0:
            raise ValueError(
                "GGUF vocabulary and tensor counts must be positive integers"
            )
    types = value["tensor_types"]
    if (
        not isinstance(types, Mapping)
        or not types
        or set(types) - {"F32", "F16", "Q8_0"}
    ):
        raise ValueError("GGUF tensor storage inventory is missing or unsupported")
    if (
        any(type(count) is not int or count <= 0 for count in types.values())
        or sum(types.values()) != value["tensor_count"]
    ):
        raise ValueError("GGUF tensor storage counts are inconsistent")
    return value


def validate_conversion(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "format",
        "quantizer_sha256",
        "source",
        "candidate",
    }:
        raise ValueError("a complete pinned GGUF conversion manifest is required")
    if value["schema"] != CONVERSION_SCHEMA or value["format"] != "Q8_0":
        raise ValueError(
            "this verification scope supports the pinned GGUF Q8_0 conversion method"
        )
    _pin(value["quantizer_sha256"])
    for role in ("source", "candidate"):
        row = value[role]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"sha256", "gguf"}
            or not isinstance(row["gguf"], Mapping)
        ):
            raise ValueError(
                "each converted model requires its content pin and GGUF inspection"
            )
        _pin(row["sha256"])
        _inspection(row["gguf"])
    left, right = value["source"], value["candidate"]
    if left["sha256"] == right["sha256"]:
        raise ValueError(
            "quantization requires distinct source and candidate artifacts"
        )
    a, b = left["gguf"], right["gguf"]
    for field in ("tokenizer_sha256", "controls_sha256", "tensor_layout_sha256"):
        if _pin(a.get(field)) != _pin(b.get(field)):
            raise ValueError(
                "quantization must preserve tokenizer, nonquantization metadata and tensor shapes"
            )
    if a.get("file_type") not in (None, 0, 1) or set(a.get("tensor_types", {})) - {
        "F32",
        "F16",
    }:
        raise ValueError("quantization reference must have unquantized F32/F16 storage")
    if (
        b.get("file_type") != 7
        or b.get("quantization_version") != 2
        or not b.get("tensor_types", {}).get("Q8_0")
    ):
        raise ValueError(
            "candidate must have observed Q8_0 tensors and quantization version 2"
        )
    return value


def _validate(recipe: StudyRecipe, registries: Any) -> Mapping[str, Any]:
    if len(recipe.study.runs) != 2:
        raise ValueError(
            "quantization verification requires reference and candidate runs"
        )
    comparison = recipe.study.comparison
    if comparison.vary != QUANTIZATION_VARIATIONS or comparison.waivers:
        raise ValueError(
            "quantization permits only its declared artifact and storage differences, without waivers"
        )
    if comparison.analyses not in {
        (TrajectoryAgreementAnalysis.name,),
        (TrajectoryAgreementAnalysis.name, VerificationImpactAnalysis.name),
    }:
        raise ValueError(
            "quantization verification requires registered trajectory analysis"
        )
    if set(recipe.environment) != {CAPTURE_KEY, QUANTIZATION_KEY}:
        raise ValueError(
            "quantization requires a qualified provider pin and conversion manifest"
        )
    _pin(recipe.environment[CAPTURE_KEY])
    conversion = validate_conversion(recipe.environment[QUANTIZATION_KEY])
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError("quantization verification requires one trajectory workload")
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    for index, (role, run) in enumerate(
        zip(("source", "candidate"), recipe.study.runs, strict=True)
    ):
        pin = validate_cpu_run(run, index, config, registries)
        if pin != conversion[role]["sha256"]:
            raise ValueError("run model does not match its conversion artifact pin")
        directory = run.runtime.get("bin_dir")
        if (
            not isinstance(directory, str)
            or str(Path(directory).expanduser().resolve()) != directory
        ):
            raise ValueError(
                "quantization requires an explicit canonical provider directory"
            )
        model = run.model.get("path")
        if (
            not isinstance(model, str)
            or str(Path(model).expanduser().resolve()) != model
        ):
            raise ValueError("quantization requires canonical model artifact paths")
    left, right = recipe.study.runs
    if left.runtime != right.runtime or left.scenario != right.scenario:
        raise ValueError(
            "quantization requires identical runtime and generation controls"
        )
    if {k: v for k, v in left.model.items() if k not in {"path", "sha256"}} != {
        k: v for k, v in right.model.items() if k not in {"path", "sha256"}
    }:
        raise ValueError("only the materialized model path and content pin may differ")
    if left.model["path"] == right.model["path"]:
        raise ValueError("quantized and reference model paths must be distinct")
    return conversion


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    from .verification import (
        _builtin_registries,
        _evidence_gaps,
        _legacy_performance,
        _mapping,
        _observed_facts,
    )

    measurement = CheckedTrajectoryProtocol()
    registries = replace(
        _builtin_registries(), measurements={measurement.name: measurement}
    )
    conversion = _validate(recipe, registries)
    provider = recipe.environment[CAPTURE_KEY]
    expected = {
        conversion[role]["sha256"]: conversion[role]["gguf"]
        for role in ("source", "candidate")
    }
    quality = task_check_identity(recipe.measurement_configs[measurement.name])

    def gaps(record: RunRecord) -> tuple[str, ...]:
        missing = _evidence_gaps(record, provider) + task_check_gaps(record, quality)
        gguf = _mapping(record.observed.get("gguf"))
        if gguf != expected[record.requested.model["sha256"].lower()]:
            missing += (
                "observed model storage or tokenizer differs from the pinned conversion",
            )
        if _observed_facts(record).get("vocab_size") != gguf.get("vocab_size"):
            missing += ("native vocabulary size differs from the embedded tokenizer",)
        return missing

    def facts(record: RunRecord) -> dict[str, Any]:
        gguf = _mapping(record.observed.get("gguf"))
        return {
            **_observed_facts(record),
            "tensor_types": gguf.get("tensor_types"),
            "tokenizer_sha256": gguf.get("tokenizer_sha256"),
        }

    directory = recipe.study.runs[0].runtime["bin_dir"]
    return VerificationRoute(
        scope=GGUF_QUANTIZATION_SCOPE,
        adapters={
            "llamacpp": QuantizedLlamaCppAdapter({directory: provider}, expected)
        },
        measurements=registries.measurements,
        analyses=registries.analyses,
        evidence_gaps=gaps,
        observed_facts=facts,
        performance=_legacy_performance,
        change={
            "parameter": "model.weight_storage",
            "format": "Q8_0",
            "reference": conversion["source"]["gguf"]["tensor_types"],
            "candidate": conversion["candidate"]["gguf"]["tensor_types"],
        },
    )
