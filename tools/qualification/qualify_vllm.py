"""Qualify pinned vLLM plain-completion capture on an explicitly selected device."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metria import ArtifactManifest, RunSpec, RunStatus, execute_run
from metria.artifacts import artifact_to_data, verify_artifact
from metria.hardware import capture_hardware_fingerprint
from metria.measurements import TokenTrajectoryProtocol, compare_trajectory_results
from metria.processes import run_process
from metria.protocols import MeasurementResult
from metria.recipes import _json_value
from metria.records import run_record_from_data, run_record_to_json
from metria.runtimes.vllm import VLLMAdapter

_WHEELS = {
    "cpu": "0ee75278b3626c5d0b7c310c6d62afae93e900f4eac339c91e333fde5108ed78",
    "cuda": "e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc",
}


class ResetCapture(TokenTrajectoryProtocol):
    """Exercise actual reset with the existing trajectory measurement contract."""

    def execute(self, session, scenario, config) -> MeasurementResult:
        first = super().execute(session, scenario, config)
        session.reset("measurement")
        second = super().execute(session, scenario, config)
        compared = compare_trajectory_results(first, second)
        if not compared.evidence["all_trajectories_match"]:
            raise RuntimeError("greedy trajectories changed after reset")
        return replace(
            second,
            evidence={
                **second.evidence,
                "qualification_reset": {
                    "scope": "measurement",
                    "repeated_trajectories_match": True,
                },
            },
        )


def _redact_paths(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for path, label in replacements.items():
            value = value.replace(path, label)
        return value
    if isinstance(value, dict):
        return {key: _redact_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item, replacements) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(_WHEELS), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.source_revision) is None:
        parser.error("source-revision must be an immutable Metria commit")
    model_root = args.model_root.resolve()
    descriptor = json.loads(
        Path(__file__).with_name("vllm-smollm2-135m.json").read_text()
    )
    model_artifacts = []
    for name, digest in descriptor["files"].items():
        manifest = ArtifactManifest(
            name=name,
            kind="model-or-tokenizer",
            sha256=digest,
            revision=descriptor["revision"],
            source={
                "repository": descriptor["model_id"],
                "license": descriptor["license"],
            },
        )
        verified = verify_artifact(
            manifest, (model_root / name).resolve(), max_bytes=300_000_000
        )
        model_artifacts.append(artifact_to_data(verified))
    runtime = verify_artifact(
        ArtifactManifest(
            name="vllm", kind="runtime-wheel", sha256=_WHEELS[args.backend]
        ),
        args.runtime_wheel,
        max_bytes=600_000_000,
    )
    version = "0.30.0+cu129" if args.backend == "cuda" else "0.30.0+cpu"
    if (
        importlib.metadata.version("vllm") != version
        or importlib.metadata.version("setuptools") != "84.0.0"
    ):
        parser.error(
            "use the pinned runtime with the recorded setuptools 84.0.0 override"
        )
    import torch

    hardware = dict(capture_hardware_fingerprint().to_mapping())
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            parser.error(
                "CUDA is unavailable; device visibility alone is not qualification"
            )
        properties = torch.cuda.get_device_properties(0)
        driver = run_process(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout_s=10,
        )
        hardware["accelerators"] = [
            {
                "source": "torch.cuda.get_device_properties and nvidia-smi",
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "memory_total_bytes": properties.total_memory,
                "driver_version": driver.stdout.strip(),
                "cuda_runtime": torch.version.cuda,
            }
        ]
    spec = RunSpec(
        model={
            "id": descriptor["model_id"],
            "path": str(model_root),
            "revision": descriptor["revision"],
            "tokenizer_revision": descriptor["revision"],
        },
        runtime={
            "name": "vllm",
            "version": version,
            "dtype": "float16" if args.backend == "cuda" else "float32",
            "max_model_len": 128,
            "max_num_seqs": 1,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.65,
        },
        scenario={
            "context": 96,
            "max_tokens": 8,
            "temperature": 0.0,
            "seed": 42,
            "chat_template": False,
        },
        measurements=(TokenTrajectoryProtocol.name,),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    record = execute_run(
        study_name=f"vllm-{args.backend}-qualification",
        run_id="smollm2-135m",
        spec=spec,
        adapter=VLLMAdapter(),
        measurement=ResetCapture(),
        measurement_config={
            "prompts": [
                {"id": "capital", "prompt": "The capital of France is"},
                {"id": "arithmetic", "prompt": "Two plus two equals"},
            ]
        },
        environment={},
    )
    qualification = {
        "schema": "metria.runtime_qualification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": args.source_revision,
        "scope": f"vLLM {version}; {args.backend}; two plain greedy prompts and reset; no performance or general quality claim",
        "hardware": hardware,
        "runtime_artifact": artifact_to_data(runtime),
        "model_artifacts": model_artifacts,
        "packages": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        },
        "dependency_override": {
            "setuptools": "84.0.0 replaces upstream's older build-tool constraint to meet repository security minimum"
        },
        "path_redactions": [
            "MODEL_SNAPSHOT",
            "MODEL_CACHE",
            "RUNTIME_ENV",
            "RUNTIME_INSTALL",
            "METRIA_SOURCE",
        ],
        "staleness": "Rerun on any Metria adapter, runtime/model/tokenizer/hash, dependency, driver, hardware, or qualification recipe change; review after 90 days.",
    }
    record = replace(
        record, provenance={**record.provenance, "qualification": qualification}
    )
    replacements = {
        str(model_root): "${MODEL_SNAPSHOT}",
        str(Path(sys.prefix)): "${RUNTIME_ENV}",
        str(Path(__file__).resolve().parents[2]): "${METRIA_SOURCE}",
        str(args.runtime_wheel.resolve().parent): "${RUNTIME_INSTALL}",
    }
    if (
        model_root.parent.name == "snapshots"
        and model_root.parent.parent.name.startswith("models--")
    ):
        replacements[str(model_root.parent.parent)] = "${MODEL_CACHE}"
    data = _redact_paths(json.loads(run_record_to_json(record)), replacements)
    # Round-trip the public record so all schema/privacy transformations remain valid.
    normalized = run_record_from_data(data)
    output = args.output / "qualification.run.json"
    output.write_text(run_record_to_json(normalized) + "\n", encoding="utf-8")
    summary = {
        "status": record.status.value,
        "record": {
            "path": output.name,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
        "qualification": _redact_paths(
            _json_value(qualification, path="qualification"), replacements
        ),
    }
    (args.output / "qualification.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": record.status.value, "backend": args.backend}))
    return 0 if record.status is RunStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
