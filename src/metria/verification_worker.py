"""Bound native verification in one child process while retaining partial records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .comparison import compare_runs
from .hardware import capture_hardware_fingerprint
from .models import RunRecord, RunStatus
from .processes import ProcessError, ProcessResult, run_process
from .recipes import (
    StudyRecipe,
    load_study_recipe,
    study_recipe_digest,
    study_recipe_to_json,
)
from .records import (
    load_run_record,
    run_evidence_digest,
    run_record_digest,
    run_record_to_json,
)
from .verification_route import VerificationRoute
from .verification_schema import VERIFICATION_ROLES, VERIFICATION_SCHEMA

_TRANSPORT = ".worker-recipe.json"
_RECEIPT = "worker-result.json"


def _read_receipt(
    output: Path, recipe: StudyRecipe, route: VerificationRoute, result: ProcessResult
) -> dict[str, Any]:
    from .verification import VERIFICATION_EXIT_CODES

    path = output / _RECEIPT
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("worker result exceeded its supported size")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data["schema"] != VERIFICATION_SCHEMA
        or data["scope"] != route.scope
        or data["recipe_digest"] != study_recipe_digest(recipe)
    ):
        raise ValueError("worker result has an invalid identity")
    code = data["exit_code"]
    if (
        isinstance(code, bool)
        or not isinstance(code, int)
        or code not in {0, 1, 2, 3, 4, 5, 130}
        or code != result.returncode
    ):
        raise ValueError("worker outcome does not match its process status")
    expected = VERIFICATION_EXIT_CODES.get(data.get("verdict"))
    if expected is None or (
        code != expected and not (code == 130 and data["verdict"] == "EXECUTION_FAILED")
    ):
        raise ValueError("worker verdict and exit code disagree")
    for index, role in enumerate(VERIFICATION_ROLES):
        entry = data["records"][role]
        if entry["path"] != f"{role}.run.json":
            raise ValueError("worker record path is invalid")
        record = load_run_record(output / entry["path"])
        if (
            record.requested != recipe.study.runs[index]
            or run_record_digest(record) != entry["record_digest"]
            or run_evidence_digest(record) != entry["evidence_digest"]
        ):
            raise ValueError("worker record identity or digest is invalid")
        if entry["status"] != record.status.value or (
            code in {0, 1} and record.status is not RunStatus.COMPLETED
        ):
            raise ValueError(
                "worker success is inconsistent with retained lifecycle evidence"
            )
    return data


def _partial_records(
    output: Path, recipe: StudyRecipe, status: RunStatus
) -> list[RunRecord]:
    from .verification import _write_atomic

    records = []
    first_missing = True
    for index, role in enumerate(VERIFICATION_ROLES):
        path = output / f"{role}.run.json"
        if path.exists():
            record = load_run_record(path)
            if record.requested != recipe.study.runs[index]:
                raise ValueError("partial worker record does not belong to this recipe")
        else:
            record = RunRecord(
                study_name=recipe.study.name,
                run_id=f"run-{index:04d}",
                requested=recipe.study.runs[index],
                resolved={},
                observed={},
                status=status if first_missing else RunStatus.CANCELLED,
                events=(
                    {
                        "stage": "verification_worker",
                        "kind": status.value if first_missing else "cancelled",
                    },
                ),
            )
            first_missing = False
            _write_atomic(path, run_record_to_json(record) + "\n")
        records.append(record)
    return records


def _failure_result(
    output: Path,
    recipe: StudyRecipe,
    route: VerificationRoute,
    failure: str,
    *,
    interrupted: bool,
    timed_out: bool,
) -> dict[str, Any]:
    from . import __version__
    from .verification import _comparison_data

    status = (
        RunStatus.INTERRUPTED
        if interrupted
        else RunStatus.TIMED_OUT
        if timed_out
        else RunStatus.FAILED
    )
    records = _partial_records(output, recipe, status)
    return {
        "schema": VERIFICATION_SCHEMA,
        "scope": route.scope,
        "study": recipe.study.name,
        "recipe_digest": study_recipe_digest(recipe),
        "implementation": {"name": "metria.verify", "version": __version__},
        "hardware": capture_hardware_fingerprint().to_mapping(),
        "verdict": "EXECUTION_FAILED",
        "exit_code": 130 if interrupted else 5,
        "acceptance_policy_evaluated": False,
        "lifecycle": {
            "status": "failed",
            "records": {
                role: record.status.value
                for role, record in zip(VERIFICATION_ROLES, records, strict=True)
            },
        },
        "comparison_status": "NOT_EVALUATED",
        "policy_status": "NOT_EVALUATED" if recipe.policy else "NOT_CONFIGURED",
        "change": route.change,
        "comparison_plan": {
            "vary": sorted(recipe.study.comparison.vary),
            "control": sorted(recipe.study.comparison.control),
            "block_by": sorted(recipe.study.comparison.block_by),
        },
        "records": {
            role: {
                "path": f"{role}.run.json",
                "run_id": record.run_id,
                "status": record.status.value,
                "record_digest": run_record_digest(record),
                "evidence_digest": run_evidence_digest(record),
                "evidence_gaps": [
                    "the bounded verification worker did not complete",
                    *route.evidence_gaps(record),
                ],
                "observed": route.observed_facts(record),
            }
            for role, record in zip(VERIFICATION_ROLES, records, strict=True)
        },
        "comparison": _comparison_data(
            compare_runs(records[0], records[1], recipe.study.comparison)
        ),
        "controls": [],
        "analyses": [],
        "systems": {role: {"available": False} for role in VERIFICATION_ROLES},
        "performance": {
            "available": False,
            "reason": "bounded execution did not complete",
        },
        "execution_error": {"error_type": failure},
    }


def _invoke_worker(recipe: StudyRecipe, output: Path) -> ProcessResult:
    path = output / _TRANSPORT
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(study_recipe_to_json(recipe))
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
        )
        return run_process(
            [
                sys.executable,
                "-m",
                "metria.verification_worker",
                "--recipe",
                str(path),
                "--output",
                str(output),
                "--digest",
                study_recipe_digest(recipe),
            ],
            timeout_s=float(recipe.environment["verification_timeout_s"]),
            env=environment,
            max_output_bytes=1024 * 1024,
        )
    finally:
        path.unlink(missing_ok=True)


def verify_isolated(
    recipe: StudyRecipe, output_dir: str | Path, route: VerificationRoute
):
    from .verification import VerificationResult, _persist_verification

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    process = None
    interrupted = False
    failure = "WorkerFailure"
    data = None
    try:
        process = _invoke_worker(recipe, output)
        if not process.timed_out:
            data = _read_receipt(output, recipe, route, process)
        else:
            failure = "WorkerTimeout"
    except KeyboardInterrupt:
        interrupted = True
        failure = "KeyboardInterrupt"
    except (OSError, ValueError, TypeError, KeyError, ProcessError) as exc:
        failure = type(exc).__name__
    if data is None:
        data = _failure_result(
            output,
            recipe,
            route,
            failure,
            interrupted=interrupted,
            timed_out=process is not None and process.timed_out,
        )
    data["execution_boundary"] = {
        "mode": "bounded_native_worker",
        "configured_engine_start_method": "spawn",
        "timeout_seconds": recipe.environment["verification_timeout_s"],
        "elapsed_seconds": process.elapsed_s if process else None,
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest()
        if process
        else None,
        "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest()
        if process
        else None,
        "stdout_truncated": process.stdout_truncated if process else None,
        "stderr_truncated": process.stderr_truncated if process else None,
        "log_hash_scope": "retained bounded process output",
        "includes_startup_and_cleanup": True,
    }
    (output / _RECEIPT).unlink(missing_ok=True)
    _persist_verification(output, data)
    return VerificationResult(output, data, data["exit_code"])


def main() -> int:
    from .verification import _verification_route, _verify_with_profile
    from .verification_vllm import VLLM_SCOPE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--digest", required=True)
    args = parser.parse_args()
    recipe_path, output = args.recipe.resolve(), args.output.resolve()
    if (
        recipe_path.parent != output
        or recipe_path.name != _TRANSPORT
        or not output.is_dir()
    ):
        raise ValueError("worker requires its reserved transport directory")
    recipe = load_study_recipe(recipe_path)
    if study_recipe_digest(recipe) != args.digest:
        raise ValueError("worker recipe digest mismatch")
    route = _verification_route(recipe)
    if route.scope != VLLM_SCOPE:
        raise ValueError("native worker accepts only the qualified vLLM profile")
    result = _verify_with_profile(
        recipe, output, route, reserved_output=True, defer_result=True
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
