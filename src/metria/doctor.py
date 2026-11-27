"""``metria doctor`` and ``metria init``: guided onboarding without silent trust.

``doctor`` inspects only what you point it at (an engine environment, a model
or tokenizer directory, endpoint environment-variable names, a policy file) and
reports each check as ok, warn or fail with a concrete fix. It never launches
an inference engine, downloads anything or reads secret values; endpoint
variables are checked for presence and URL shape only.

``init`` writes reviewable starter files into a new directory: an example
workload with task checks, a policy template, proposed file pins for a model or
tokenizer directory, and next-step commands. Pins are written as *proposed*
unless ``--approve-pins`` says you have reviewed them; nothing is overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from . import __version__

SCHEMA = "metria.doctor.v1"
_OPTIONAL = {
    "vllm": "local vLLM profiles (prefix, upgrade, serving, FP8, parallelism)",
    "torch": "vLLM's tensor backend",
    "tokenizers": "local tokenization for endpoint verification",
    "jinja2": "--chat-template rendering",
    "cryptography": "Ed25519 collector signatures",
}
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str | None = None

    def as_mapping(self) -> dict[str, Any]:
        row = {"name": self.name, "status": self.status, "detail": self.detail}
        if self.fix:
            row["fix"] = self.fix
        return row


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python() -> list[Check]:
    major, minor = sys.version_info[:2]
    checks = [Check("metria", "ok", f"metria {__version__}")]
    if (major, minor) < (3, 10) or (major, minor) > (3, 14):
        checks.append(
            Check("python", "fail", f"Python {major}.{minor}", "use Python 3.10-3.14")
        )
    elif (major, minor) == (3, 10):
        checks.append(
            Check(
                "python",
                "warn",
                "Python 3.10",
                "native backend extras need Python 3.11+",
            )
        )
    else:
        checks.append(
            Check("python", "ok", f"Python {major}.{minor} on {platform.system()}")
        )
    return checks


def _qualified_profiles(vllm_version: str) -> list[str]:
    from .verification_kv_precision import QUALIFIED_KV_VERSIONS
    from .verification_parallelism import QUALIFIED_PARALLEL_VERSIONS
    from .verification_vllm import QUALIFIED_VLLM_VERSIONS

    profiles = []
    if vllm_version in QUALIFIED_VLLM_VERSIONS:
        profiles += ["prefix caching", "serving and memory"]
    if vllm_version.endswith("+cpu") and vllm_version in QUALIFIED_VLLM_VERSIONS:
        profiles.append("CPU runtime upgrade")
    if vllm_version in QUALIFIED_KV_VERSIONS:
        profiles.append("FP8 KV cache")
    if vllm_version in QUALIFIED_PARALLEL_VERSIONS:
        profiles.append("tensor/pipeline parallelism (staged)")
    return profiles


def check_packages(version: Callable[[str], str | None] = _version) -> list[Check]:
    checks = []
    for name, purpose in _OPTIONAL.items():
        found = version(name)
        if found is None:
            checks.append(
                Check(f"package.{name}", "warn", f"not installed: needed for {purpose}")
            )
        else:
            checks.append(Check(f"package.{name}", "ok", f"{name} {found}"))
    vllm = version("vllm")
    if vllm is not None:
        profiles = _qualified_profiles(vllm)
        if profiles:
            checks.append(
                Check("vllm.qualified", "ok", "qualified for: " + ", ".join(profiles))
            )
        else:
            checks.append(
                Check(
                    "vllm.qualified",
                    "fail",
                    f"vllm {vllm} is not a qualified version",
                    "install vllm==0.30.0+cpu or vllm==0.30.0+cu129 (see runtime qualification)",
                )
            )
    return checks


def _run(command: list[str]) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        command, check=True, capture_output=True, text=True, timeout=20
    ).stdout


def check_gpus(run: Callable[[list[str]], str] = _run) -> list[Check]:
    try:
        output = run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]
        )
    except (OSError, subprocess.SubprocessError):
        return [
            Check(
                "gpu", "warn", "no NVIDIA GPU detected: CPU and endpoint profiles only"
            )
        ]
    rows = [line.split(",") for line in output.strip().splitlines() if line.strip()]
    if not rows:
        return [Check("gpu", "warn", "nvidia-smi listed no GPUs")]
    capabilities = [float(cap.strip()) for _, cap in rows]
    names = ", ".join(name.strip() for name, _ in rows)
    checks = [Check("gpu", "ok", f"{len(rows)} GPU(s): {names}")]
    if min(capabilities) >= 8.9:
        checks.append(
            Check("gpu.fp8", "ok", "compute capability 8.9+: FP8 KV cache supported")
        )
    else:
        checks.append(
            Check(
                "gpu.fp8",
                "warn",
                f"compute capability {min(capabilities)}: FP8 KV cache needs 8.9+",
            )
        )
    if len(rows) < 2:
        checks.append(
            Check("gpu.parallel", "warn", "one GPU: parallelism needs at least two")
        )
    return checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_dir(path: Path) -> list[Check]:
    if not path.is_dir():
        return [
            Check(
                "model",
                "fail",
                f"{path.name}: not a directory",
                "pass an existing local model or tokenizer directory",
            )
        ]
    checks = []
    if (path / "config.json").is_file():
        checks.append(Check("model.config", "ok", "config.json present"))
    else:
        checks.append(
            Check(
                "model.config",
                "warn",
                "no config.json: usable only as a tokenizer directory",
            )
        )
    if (path / "tokenizer.json").is_file():
        checks.append(Check("model.tokenizer", "ok", "tokenizer.json present"))
    else:
        checks.append(
            Check(
                "model.tokenizer",
                "fail",
                "no tokenizer.json",
                "endpoint verification tokenizes locally; point --model at the tokenizer files",
            )
        )
    return checks


def check_endpoint_env(names: Sequence[str]) -> list[Check]:
    from .runtimes.endpoint_http import parse_base_url

    checks = []
    for name in names:
        value = os.environ.get(name)
        if not value:
            checks.append(
                Check(f"env.{name}", "fail", "not set", f"export {name}=https://…")
            )
            continue
        try:
            parse_base_url(value)
        except ValueError as exc:
            checks.append(
                Check(f"env.{name}", "fail", str(exc), "use https, or http on loopback")
            )
        else:
            checks.append(
                Check(
                    f"env.{name}", "ok", "set to a valid endpoint URL (value not shown)"
                )
            )
    return checks


def check_policy(path: Path) -> list[Check]:
    from .policies import policy_from_data

    try:
        policy_from_data(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        return [
            Check(
                "policy",
                "fail",
                str(exc)[:300],
                "fix the policy file; see verification policies",
            )
        ]
    return [Check("policy", "ok", f"{path.name} is a valid policy")]


def run_doctor(args: Any) -> list[Check]:
    checks = check_python() + check_packages()
    if not args.no_gpus:
        checks += check_gpus()
    if args.model is not None:
        checks += check_model_dir(args.model)
    checks += check_endpoint_env(args.endpoint_env)
    if args.policy is not None:
        checks += check_policy(args.policy)
    return checks


def doctor_command(args: Any, stdout: TextIO) -> int:
    checks = run_doctor(args)
    failed = [c for c in checks if c.status == "fail"]
    if args.json_output:
        stdout.write(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": not failed,
                    "checks": [c.as_mapping() for c in checks],
                },
                indent=2,
            )
            + "\n"
        )
    else:
        marks = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}
        for check in checks:
            stdout.write(f"[{marks[check.status]}] {check.name}: {check.detail}\n")
            if check.fix:
                stdout.write(f"       fix: {check.fix}\n")
        stdout.write(
            "\nNo blocking problems found.\n"
            if not failed
            else f"\n{len(failed)} blocking problem(s).\n"
        )
    return 1 if failed else 0


_EXAMPLE_PROMPTS = (
    ("capital", "The capital of France is"),
    ("arithmetic", "Two plus two equals"),
    ("story", "Once upon a time"),
    ("definition", "A prime number is"),
)
_POLICY_TEMPLATE = {
    "schema": "metria.verification_policy.v1",
    "criteria": [
        {"target": "quality.candidate_pass_rate", "version": "1", "min": 1.0},
        {"target": "behavior.trajectory_agreement", "version": "0.3.4", "min": 0.9},
    ],
}


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _next_steps(args: Any, pins: str | None) -> str:
    lines = [
        "# Next steps",
        "",
        "Review every file here before using it: thresholds in policy.json and the",
        "prompts and checks in workload.jsonl are examples, not recommendations.",
        "",
    ]
    if pins and pins.endswith(".proposed.json"):
        lines += [
            f"{pins} pins the files on disk now. It proves only that they have",
            "not changed since; confirm they are the files you intend to trust (for",
            "example against the publisher's revision), then rerun with --approve-pins",
            "or rename it to descriptor.json.",
            "",
        ]
    lines += [
        "Endpoint verification (two running servers):",
        "",
        "```bash",
        "export METRIA_REFERENCE_URL=https://reference.example",
        "export METRIA_CANDIDATE_URL=https://candidate.example",
        "metria doctor --model <tokenizer-dir> --endpoint-env METRIA_REFERENCE_URL "
        "--endpoint-env METRIA_CANDIDATE_URL --policy policy.json",
        "metria recipe prepare-endpoint --model-id <served-name> --tokenizer <tokenizer-dir> \\",
        "  --workload workload.jsonl --policy policy.json \\",
        "  --reference-url-env METRIA_REFERENCE_URL --candidate-url-env METRIA_CANDIDATE_URL \\",
        "  --reference-declared kv_cache_dtype=auto --candidate-declared kv_cache_dtype=fp8 \\",
        "  --output study.json",
        "metria verify study.json --output verification",
        "```",
        "",
        "Local vLLM verification (qualified runtime environment, Linux or WSL):",
        "",
        "```bash",
        "metria doctor --model <model-dir> --policy policy.json",
        "metria recipe prepare-vllm --model <model-dir> --descriptor descriptor.json \\",
        "  --workload workload.jsonl --policy policy.json --output study.json",
        "metria verify study.json --output verification",
        "```",
        "",
        "Without a model or server, `metria demo --case pass --output demo` shows a",
        "synthetic report. It proves nothing about any runtime.",
    ]
    return "\n".join(lines) + "\n"


def init_command(args: Any, stdout: TextIO) -> int:
    target: Path = args.output_dir
    if target.exists():
        raise FileExistsError(f"{target} already exists; choose a new directory")
    pins_name = None
    pins = None
    if args.model is not None:
        if not args.model.is_dir():
            raise ValueError("--model must be an existing directory")
        files = {
            path.name: _sha256(path)
            for path in sorted(args.model.iterdir())
            if path.is_file()
            and (
                path.name in _TOKENIZER_FILES
                or path.suffix in {".json", ".safetensors"}
            )
        }
        pins = {
            "model_id": args.model.resolve().name,
            "files": files,
            "note": "SHA-256 pins computed by metria init from local files; "
            "review them before trusting this descriptor",
        }
        pins_name = (
            "descriptor.json" if args.approve_pins else "descriptor.proposed.json"
        )
    target.mkdir(parents=True)
    workload = "".join(
        json.dumps(
            {
                "id": pid,
                "prompt": text,
                "checks": [{"id": "answered", "kind": "nonempty"}],
            }
        )
        + "\n"
        for pid, text in _EXAMPLE_PROMPTS
    )
    _write_new(target / "workload.jsonl", workload)
    _write_new(target / "policy.json", json.dumps(_POLICY_TEMPLATE, indent=2) + "\n")
    if pins is not None and pins_name is not None:
        _write_new(
            target / pins_name, json.dumps(pins, indent=2, sort_keys=True) + "\n"
        )
    _write_new(target / "NEXT_STEPS.md", _next_steps(args, pins_name))
    written = sorted(p.name for p in target.iterdir())
    stdout.write(f"Wrote {', '.join(written)} to {target}\nRead NEXT_STEPS.md first.\n")
    return 0


def add_onboarding_parsers(subparsers: Any) -> None:
    doctor = subparsers.add_parser(
        "doctor", help="check this environment, inputs and endpoint variables"
    )
    doctor.add_argument("--model", type=Path, help="local model or tokenizer directory")
    doctor.add_argument(
        "--endpoint-env",
        action="append",
        default=[],
        help="endpoint URL variable name (repeatable)",
    )
    doctor.add_argument("--policy", type=Path, help="policy JSON to validate")
    doctor.add_argument("--no-gpus", action="store_true", help="skip nvidia-smi")
    doctor.add_argument("--json", dest="json_output", action="store_true")
    init = subparsers.add_parser("init", help="write reviewable starter files")
    init.add_argument("--output-dir", type=Path, required=True)
    init.add_argument(
        "--model", type=Path, help="model or tokenizer directory to pin in a descriptor"
    )
    init.add_argument(
        "--approve-pins",
        action="store_true",
        help="write descriptor.json instead of descriptor.proposed.json",
    )
