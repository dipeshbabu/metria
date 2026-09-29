"""Bounded local interpreter execution with request and evidence binding."""

from __future__ import annotations

import hashlib
import secrets
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ._freeze import freeze_mapping
from .capability_checks import CapabilityCheckRegistry
from .measurements.prefix_workload import PrefixWorkloadProtocol
from .models import RunRecord, RunSpec, RunStatus
from .processes import ProcessResult, run_process
from .protocols import MeasurementProtocol, RuntimeAdapter
from .recipes import _json_value, run_spec_to_data
from .records import run_record_digest, run_record_from_data
from .runtime_environment import (
    REQUEST_SCHEMA,
    child_environment,
    python_path,
    read_packet,
    source_digest,
    validate_receipt,
    write_private,
)
from .runtimes.llamacpp import _sha256_file


def _failure(
    study: str,
    run_id: str,
    spec: RunSpec,
    status: RunStatus,
    reason: str,
    result: ProcessResult | None = None,
    previous: RunRecord | None = None,
) -> RunRecord:
    details: dict[str, Any] = {"status": status.value, "reason": reason}
    if result is not None:
        details.update(
            {
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            }
        )
    record = previous or RunRecord(
        study_name=study,
        run_id=run_id,
        requested=spec,
        resolved={},
        observed={},
        status=status,
    )
    return replace(
        record,
        status=status,
        evidence={**record.evidence, "external_execution": details},
        events=(*record.events, {"stage": "runtime_environment", "kind": reason}),
    )


def _bound_record(
    root: Path,
    request_digest: str,
    request: Mapping[str, Any],
    target: Mapping[str, Any],
    spec: RunSpec,
) -> RunRecord:
    receipt = read_packet(root / "receipt.json")
    validate_receipt(receipt, nonce=request["nonce"], request_sha256=request_digest)
    if receipt.get("environment") != _json_value(target, path="runtime_environment"):
        raise ValueError("worker observed a different runtime environment")
    record = run_record_from_data(read_packet(root / "record.json"))
    if receipt.get("record_digest") != run_record_digest(record):
        raise ValueError("worker run record changed after receipt creation")
    if (record.study_name, record.run_id, record.requested) != (
        request["study_name"],
        request["run_id"],
        spec,
    ):
        raise ValueError("worker evidence belongs to another requested run")
    for location in (record.resolved, record.observed):
        if location.get("runtime_environment") != target:
            raise ValueError("run record lacks its independently observed environment")
    return record


class RuntimeRunExecutor:
    """Execute the same first-party lifecycle in each pinned local interpreter."""

    def __init__(
        self,
        targets: Mapping[str, Any],
        *,
        timeout_s: float,
        binding: str,
        cpu_ids: tuple[int, int],
    ) -> None:
        self._targets = freeze_mapping(targets)
        self._timeout = timeout_s
        self._binding = binding
        self._cpu_ids = cpu_ids
        self._deadline: float | None = None

    def __call__(
        self,
        *,
        study_name: str,
        run_id: str,
        spec: RunSpec,
        adapter: RuntimeAdapter,
        measurement: MeasurementProtocol,
        measurement_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        capability_checks: CapabilityCheckRegistry | None = None,
    ) -> RunRecord:
        if (
            adapter.name != "vllm"
            or type(measurement) is not PrefixWorkloadProtocol
            or capability_checks is not None
        ):
            raise ValueError(
                "isolated runtime upgrades use only their built-in lifecycle and checks"
            )
        if self._deadline is None:
            self._deadline = time.monotonic() + self._timeout
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            return _failure(
                study_name,
                run_id,
                spec,
                RunStatus.TIMED_OUT,
                "verification_deadline_exhausted",
            )
        target = self._targets[spec.environment_selector["runtime_environment"]]
        try:
            return self._execute(
                study_name, run_id, spec, measurement_config, target, remaining
            )
        except (OSError, TypeError, ValueError):
            return _failure(
                study_name,
                run_id,
                spec,
                RunStatus.FAILED,
                "runtime_environment_or_transport_invalid",
            )

    def _execute(
        self,
        study: str,
        run_id: str,
        spec: RunSpec,
        config: Mapping[str, Any],
        target: Mapping[str, Any],
        timeout: float,
    ) -> RunRecord:
        executable = python_path(target["python"])
        if _sha256_file(executable) != target["python_sha256"]:
            raise ValueError("runtime interpreter changed after preparation")
        request = {
            "schema": REQUEST_SCHEMA,
            "mode": "run",
            "nonce": secrets.token_hex(24),
            "metria_sha256": source_digest(),
            "target": target,
            "study_name": study,
            "run_id": run_id,
            "spec": run_spec_to_data(spec),
            "measurement_config": config,
            "cpu_binding": self._cpu_ids,
        }
        with tempfile.TemporaryDirectory(prefix="metria-runtime-run-") as directory:
            root = Path(directory).resolve()
            if root.parent != Path(tempfile.gettempdir()).resolve():
                raise ValueError("runtime worker directory escaped its temporary root")
            digest = write_private(root / "request.json", request)
            result = run_process(
                [
                    str(executable),
                    "-I",
                    "-m",
                    "metria.runtime_worker",
                    str(root / "request.json"),
                ],
                cwd=root,
                env=child_environment(
                    executable, descriptor=target, binding=self._binding
                ),
                timeout_s=timeout,
                max_output_bytes=2 * 1024 * 1024,
            )
            try:
                record = _bound_record(root, digest, request, target, spec)
            except (OSError, TypeError, ValueError):
                record = None
            if result.timed_out or result.returncode != 0:
                status = RunStatus.TIMED_OUT if result.timed_out else RunStatus.FAILED
                return _failure(
                    study,
                    run_id,
                    spec,
                    status,
                    "runtime_worker_did_not_finish",
                    result,
                    record,
                )
            if record is None:
                return _failure(
                    study,
                    run_id,
                    spec,
                    RunStatus.FAILED,
                    "runtime_worker_receipt_invalid",
                    result,
                )
            return replace(
                record,
                evidence={
                    **record.evidence,
                    "external_execution": {
                        "status": "completed",
                        "request_sha256": digest,
                        "stdout_sha256": hashlib.sha256(
                            result.stdout.encode()
                        ).hexdigest(),
                        "stderr_sha256": hashlib.sha256(
                            result.stderr.encode()
                        ).hexdigest(),
                        "stdout_truncated": result.stdout_truncated,
                        "stderr_truncated": result.stderr_truncated,
                    },
                },
            )
