"""Exercise the full verifier with deterministic fake subprocess evidence only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from metria import (
    ComparisonPlan,
    HardwareFingerprint,
    PolicyCriterion,
    RunSpec,
    StudyRecipe,
    StudySpec,
    VerificationPolicy,
    dump_study_recipe,
    verification,
)
from metria.cli import main as verify_main
from metria.measurements import TokenTrajectoryProtocol, TrajectoryAgreementAnalysis
from metria.runtimes import llamacpp


def run_fixture(case: str, output: Path) -> int:
    """Keep fakes inside this explicit test harness; the public CLI stays unchanged."""
    with tempfile.TemporaryDirectory(prefix="metria-example-") as temporary:
        root = Path(temporary)
        binary = root / (
            "llama-completion.exe" if os.name == "nt" else "llama-completion"
        )
        binary.write_bytes(b"synthetic fixture; never executable runtime evidence")
        model = root / "fixture.gguf"
        model.write_bytes(b"synthetic model fixture")
        runs = tuple(
            RunSpec(
                model={
                    "path": str(model),
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                },
                runtime={
                    "name": "llamacpp",
                    "bin_dir": str(root),
                    "threads": threads,
                    "threads_batch": 1,
                    "n_gpu_layers": 0,
                    "flash_attention": False,
                },
                scenario={
                    "context": 256,
                    "max_tokens": 3,
                    "temperature": 0.0,
                    "seed": 42,
                    "chat_template": False,
                },
                measurements=(TokenTrajectoryProtocol.name,),
            )
            for threads in (1, 2)
        )
        recipe = StudyRecipe(
            study=StudySpec(
                name="synthetic-example",
                runs=runs,
                comparison=ComparisonPlan(
                    vary=frozenset(
                        {
                            "runtime.threads",
                            "resolved.runtime.threads",
                            "observed.runtime.threads",
                            "observed.identity.applied.fields.threads",
                        }
                    ),
                    control=frozenset({"model", "scenario", "measurements"}),
                    analyses=(TrajectoryAgreementAnalysis.name,),
                ),
            ),
            measurement_configs={
                TokenTrajectoryProtocol.name: {
                    "prompts": [
                        {
                            "id": "example-one",
                            "prompt": "Synthetic fixture prompt",
                            "category": "fixture",
                        }
                    ]
                }
            },
            environment={
                "llama_cpp_token_ids_capture_sha256": hashlib.sha256(
                    binary.read_bytes()
                ).hexdigest()
            },
            policy=VerificationPolicy(
                (
                    PolicyCriterion(
                        "behavior.trajectory_agreement",
                        "0.3.4",
                        minimum=0.9 if case == "fail" else 0.6,
                    ),
                )
            ),
        )
        recipe_path = root / "study.json"
        dump_study_recipe(recipe_path, recipe)

        def fake_completion(argv, **kwargs):
            threads = int(argv[argv.index("-t") + 1])
            trajectory = Path(kwargs["env"]["KV_FIDELITY_TRAJECTORY"])
            tokens = [11, 12, 13 if threads == 1 else 14]
            trajectory.write_text(
                "".join(
                    json.dumps({"step": i, "token_id": token}) + "\n"
                    for i, token in enumerate(tokens)
                ),
                encoding="utf-8",
            )
            readback = {
                "schema": "metria.llamacpp_capture.v1",
                "threads": threads,
                "threads_batch": 1,
                "context": 256,
                "vocab_size": 1024
                if case == "not-comparable" and threads == 2
                else 512,
                "chat_template_applied": False,
            }
            Path(str(trajectory) + ".runtime.json").write_text(
                json.dumps(readback), encoding="utf-8"
            )
            return subprocess.CompletedProcess(
                argv, 0, stdout="synthetic output", stderr=""
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(llamacpp.subprocess, "run", fake_completion),
            patch.object(llamacpp, "time", SimpleNamespace(monotonic=lambda: 1.0)),
            patch.object(
                verification,
                "capture_hardware_fingerprint",
                lambda: HardwareFingerprint(platform={"system": "synthetic-fixture"}),
            ),
            patch.object(verification, "VERIFICATION_SCOPE", "synthetic_fixture.v1"),
        ):
            code = verify_main(
                ["verify", str(recipe_path), "--output", str(output), "--json"],
                stdout=stdout,
                stderr=stderr,
            )
        if not (output / "verification.json").is_file():
            raise RuntimeError(
                "fixture failed before producing a canonical result: "
                + stderr.getvalue()
            )
        data = json.loads(stdout.getvalue())
        data["scope"] = "synthetic_fixture.v1"
        data["fixture_only"] = True
        serialized = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for filename in ("verification.json", "manifest.json"):
            (output / filename).write_text(serialized, encoding="utf-8")
        (output / "report.md").write_text(
            verification.render_verification(data), encoding="utf-8"
        )
        print(verification.render_verification(data))
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("pass", "fail", "not-comparable"), default="pass"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return run_fixture(args.case, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
