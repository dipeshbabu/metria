"""STAGED: native vLLM observations for FP8 KV cache and staged treatments.

Each ``run`` invocation launches exactly one engine through Metria's real vLLM
adapter and study lifecycle, then writes one run record. ``compare`` derives the
reference/candidate trajectory comparison from saved records. Output records keep
local paths redacted. This produces engineering evidence; it does not by itself
promote a verification profile.

``--distribution`` captures staged raw top-20 logprobs with
``TokenDistributionProtocol`` instead of token IDs only, so ``compare`` also
reports trajectory KL divergence. It requires the ``metria.staged_captures``
capability override, which this tool records in the run's trial policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from metria import ArtifactManifest, RunSpec, TreatmentSpec, TreatmentType, execute_run
from metria.artifacts import artifact_to_data, verify_artifact
from metria.hardware import capture_hardware_fingerprint
from metria.integrations.staged_captures import STAGED_CAPTURE_CHECK
from metria.measurements import (
    TokenDistributionProtocol,
    TokenTrajectoryProtocol,
    compare_trajectory_distributions,
    compare_trajectory_results,
)
from metria.protocols import MeasurementResult
from metria.recipes import _json_value
from metria.records import load_run_record, run_record_from_data, run_record_to_json
from metria.runtimes.vllm import VLLMAdapter

# backend -> (pinned wheel SHA-256, runtime version, model dtype)
BACKENDS = {
    "cuda": (
        "e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc",
        "0.30.0+cu129",
        "float16",
    ),
    "cpu": (
        "0ee75278b3626c5d0b7c310c6d62afae93e900f4eac339c91e333fde5108ed78",
        "0.30.0+cpu",
        "float32",
    ),
}
PROMPTS = [
    {"id": "story", "prompt": "Once upon a time"},
    {"id": "dog", "prompt": "The little dog"},
    {"id": "capital", "prompt": "The capital of France is"},
    {"id": "arithmetic", "prompt": "Two plus two equals"},
]
VARIANTS: dict[str, dict[str, Any]] = {
    "reference": {"kv": "auto", "staged": ()},
    # Identical configuration in a fresh engine: KL divergence must be zero.
    "repeat": {"kv": "auto", "staged": ()},
    # Numeric change with a known direction: lower model precision.
    "bf16": {"kv": "auto", "staged": (), "dtype": "bfloat16"},
    "fp8": {"kv": "fp8", "staged": ()},
    "staged": {
        "kv": "auto",
        "staged": (
            (
                "vllm.scheduler",
                {"enable_chunked_prefill": True, "max_num_batched_tokens": 2048},
            ),
            ("vllm.cache_layout", {"block_size": 32}),
        ),
    },
    # Negative case: without chunked prefill a batch budget below the model
    # length cannot schedule a full prompt, so the engine must refuse it.
    "staged-reject": {
        "kv": "auto",
        "staged": (
            (
                "vllm.scheduler",
                {"enable_chunked_prefill": False, "max_num_batched_tokens": 64},
            ),
        ),
    },
}


def _redact(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for path, label in replacements.items():
            value = value.replace(path, label)
        return value
    if isinstance(value, dict):
        return {key: _redact(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, replacements) for item in value]
    return value


def _hardware(backend: str) -> dict[str, Any]:
    import torch

    hardware = dict(capture_hardware_fingerprint().to_mapping())
    if backend == "cpu":
        return hardware
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    hardware["accelerators"] = [
        {
            "source": "torch.cuda.get_device_properties and nvidia-smi",
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "memory_total_bytes": props.total_memory,
            "driver_version": driver,
            "cuda_runtime": torch.version.cuda,
        }
    ]
    return hardware


def _verified_inputs(
    args: argparse.Namespace, model_root: Path, descriptor: dict[str, Any]
) -> tuple[list[Any], Any]:
    """Verify every pinned model file and the runtime wheel before launch."""

    models = [
        artifact_to_data(
            verify_artifact(
                ArtifactManifest(
                    name=name,
                    kind="model-or-tokenizer",
                    sha256=digest,
                    revision=descriptor["revision"],
                    source={
                        "repository": descriptor["model_id"],
                        "license": descriptor["license"],
                    },
                ),
                (model_root / name).resolve(),
                max_bytes=300_000_000,
            )
        )
        for name, digest in descriptor["files"].items()
    ]
    wheel = verify_artifact(
        ArtifactManifest(
            name="vllm", kind="runtime-wheel", sha256=BACKENDS[args.backend][0]
        ),
        args.runtime_wheel,
        max_bytes=600_000_000,
    )
    return models, wheel


def _spec(
    variant: dict[str, Any],
    model_root: Path,
    descriptor: dict[str, Any],
    backend: str = "cuda",
    distribution: bool = False,
) -> RunSpec:
    _, version, dtype = BACKENDS[backend]
    dtype = variant.get("dtype", dtype)
    overrides = ["vllm.staged_treatments"] if variant["staged"] else []
    if distribution:
        overrides.append(STAGED_CAPTURE_CHECK)
    protocol = TokenDistributionProtocol if distribution else TokenTrajectoryProtocol
    treatments = [
        TreatmentSpec(
            name="vllm.kv_cache",
            kind=TreatmentType.RUNTIME_FEATURE,
            config={"dtype": variant["kv"]},
        )
    ]
    treatments += [
        TreatmentSpec(name=name, kind=TreatmentType.RUNTIME_FEATURE, config=config)
        for name, config in variant["staged"]
    ]
    return RunSpec(
        model={
            "id": descriptor["model_id"],
            "path": str(model_root),
            "revision": descriptor["revision"],
            "tokenizer_revision": descriptor["revision"],
        },
        runtime={
            "name": "vllm",
            "version": version,
            "dtype": dtype,
            "max_model_len": 128,
            "max_num_seqs": 1,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.65,
        },
        scenario={
            "context": 96,
            "max_tokens": 16,
            "temperature": 0.0,
            "seed": 42,
            "chat_template": False,
        },
        measurements=(protocol.name,),
        treatments=tuple(treatments),
        trial_policy={"capability_overrides": overrides} if overrides else {},
    )


def _run(args: argparse.Namespace) -> int:
    variant = VARIANTS[args.variant]
    model_root = args.model_root.resolve()
    descriptor = json.loads(args.descriptor.read_text(encoding="utf-8"))
    models, wheel = _verified_inputs(args, model_root, descriptor)
    record = execute_run(
        study_name="vllm-gpu-staged-observation",
        run_id=args.variant,
        spec=_spec(variant, model_root, descriptor, args.backend, args.distribution),
        adapter=VLLMAdapter(),
        measurement=(
            TokenDistributionProtocol()
            if args.distribution
            else TokenTrajectoryProtocol()
        ),
        measurement_config={"prompts": PROMPTS},
        environment={},
    )
    return _write_outputs(args, record, models, wheel, variant, model_root)


def _write_outputs(
    args: argparse.Namespace,
    record: Any,
    models: list[Any],
    wheel: Any,
    variant: dict[str, Any],
    model_root: Path,
) -> int:
    """Write the redacted run record and a plain-JSON observation summary."""

    replacements = {
        str(model_root): "${MODEL_SNAPSHOT}",
        str(Path(sys.prefix)): "${RUNTIME_ENV}",
        str(args.runtime_wheel.resolve().parent): "${RUNTIME_INSTALL}",
        str(Path(__file__).resolve().parent): "${TOOL_DIR}",
    }
    # Hugging Face snapshots symlink into the hub cache's blobs/ directory, so
    # verified file paths resolve outside the snapshot; redact the cache too.
    if (
        model_root.parent.name == "snapshots"
        and model_root.parent.parent.name.startswith("models--")
    ):
        replacements[str(model_root.parent.parent)] = "${MODEL_CACHE}"
        replacements[str(model_root.parents[2])] = "${HUB_CACHE}"
    args.output.mkdir(parents=True, exist_ok=True)
    # Serialize through the record schema first: evidence uses immutable
    # mappings that plain json.dumps cannot encode.
    data = _redact(json.loads(run_record_to_json(record)), replacements)
    out = args.output / f"{args.variant}.run.json"
    out.write_text(
        run_record_to_json(run_record_from_data(data)) + "\n", encoding="utf-8"
    )
    observed = data["record"]["observed"]
    observation = _redact(
        {
            "schema": "metria.native_observation.v1",
            "variant": args.variant,
            "status": record.status.value,
            "record_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "hardware": _json_value(_hardware(args.backend), path="observation"),
            "runtime_artifact": _json_value(
                artifact_to_data(wheel), path="observation"
            ),
            "model_artifacts": _json_value(models, path="observation"),
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            },
            "requested_kv_cache_dtype": variant["kv"],
            "requested_model_dtype": variant.get("dtype", BACKENDS[args.backend][2]),
            "measurement": data["record"]["requested"]["measurements"],
            "applied": observed.get("applied"),
            "staged_treatments": observed.get("staged_treatments"),
        },
        replacements,
    )
    (args.output / f"{args.variant}.observation.json").write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    applied = observation["applied"] or {}
    print(
        json.dumps(
            {
                "variant": args.variant,
                "status": record.status.value,
                "cache_dtype": applied.get("fields", {}).get("cache.cache_dtype"),
                "staged": observation["staged_treatments"],
            }
        )
    )
    return 0 if record.status.value == "completed" else 1


def _compare(args: argparse.Namespace) -> int:
    def capture(name: str) -> MeasurementResult:
        record = load_run_record(args.output / f"{name}.run.json")
        measurements = record.evidence["measurements"]
        if TokenDistributionProtocol.name in measurements:
            evidence = dict(measurements[TokenDistributionProtocol.name])
            distributions[name] = MeasurementResult(evidence=dict(evidence))
            # The token-ID view of a distribution capture is a trajectory.
            evidence.update(
                schema="metria.trajectory_capture.v1",
                method="kv_fidelity.decode_time_trajectory",
                method_version="0.3.4",
            )
            return MeasurementResult(evidence=evidence)
        return MeasurementResult(evidence=measurements[TokenTrajectoryProtocol.name])

    distributions: dict[str, MeasurementResult] = {}
    result = compare_trajectory_results(capture("reference"), capture(args.candidate))
    summary = {
        "schema": "metria.native_observation_comparison.v1",
        "reference": "reference",
        "candidate": args.candidate,
        "trajectory_agreement_score": result.metrics[
            "trajectory_agreement_score"
        ].value,
        "full_match_rate": result.metrics["trajectory_full_match_rate"].value,
        "all_trajectories_match": result.evidence["all_trajectories_match"],
        "per_prompt": [
            {
                k: row[k]
                for k in (
                    "id",
                    "first_divergence",
                    "prefix_agreement_steps",
                    "reference_steps",
                    "candidate_steps",
                    "matched",
                )
            }
            for row in result.evidence["per_prompt"]
        ],
    }
    if len(distributions) == 2:
        kld = compare_trajectory_distributions(
            distributions["reference"], distributions[args.candidate]
        )
        summary["trajectory_kld"] = {
            "mean_nats": kld.metrics["trajectory_kld_mean"].value,
            "max_nats": kld.metrics["trajectory_kld_max"].value,
            "coverage": kld.metrics["trajectory_kld_mean"].coverage,
            "scored_steps": kld.evidence["scored_steps"],
            "qualification": kld.evidence["qualification"],
        }
    path = args.output / f"compare-reference-{args.candidate}.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    run.add_argument("--backend", choices=sorted(BACKENDS), default="cuda")
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument("--descriptor", type=Path, required=True)
    run.add_argument("--runtime-wheel", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--distribution", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return _run(args) if args.command == "run" else _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
