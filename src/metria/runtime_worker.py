"""Private entry point for a pinned environment's normal Metria run lifecycle."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .execution import execute_run
from .measurements.prefix_workload import PrefixWorkloadProtocol
from .recipes import run_spec_from_data
from .records import run_record_digest, run_record_to_data
from .runtime_environment import (
    RECEIPT_SCHEMA,
    REQUEST_SCHEMA,
    current_environment,
    read_bound_packet,
    source_digest,
    validate_environment,
    write_private,
)
from .runtimes.vllm_placement import CPUUpgradeAdapter


def bind_cpu(cores: tuple[int, int]) -> None:
    set_affinity = getattr(os, "sched_setaffinity", None)
    if not callable(set_affinity):
        raise ValueError(
            "controlled runtime workers require Linux CPU affinity support"
        )
    set_affinity(0, set(cores))
    for task in Path("/proc/self/task").iterdir():
        try:
            set_affinity(int(task.name), set(cores))
        except ProcessLookupError:
            continue


def execute_request(request: dict[str, Any], folder: Path) -> dict[str, Any]:
    observed = current_environment()
    validate_environment(observed, metria_sha256=request["metria_sha256"])
    if request["mode"] == "inspect":
        return {"environment": observed}
    target = request["target"]
    if observed != target:
        raise ValueError("runtime environment changed after preparation")
    spec = run_spec_from_data(request["spec"])
    from .verification_upgrade import _cpu_binding, validate_upgrade_run

    validate_upgrade_run(spec, request["measurement_config"])
    cores = _cpu_binding(request["cpu_binding"], {"current": target})
    bind_cpu(cores)
    record = execute_run(
        study_name=request["study_name"],
        run_id=request["run_id"],
        spec=spec,
        adapter=CPUUpgradeAdapter(),
        measurement=PrefixWorkloadProtocol(),
        measurement_config=request["measurement_config"],
        environment={"vllm_distribution_sha256": target["runtime"]["sha256"]},
    )
    after = current_environment()
    if not set(after["cpu_affinity"] or ()) or not set(after["cpu_affinity"]) <= set(
        cores
    ):
        raise ValueError("interpreter escaped its controlled CPU placement")
    after["cpu_affinity"] = observed["cpu_affinity"]
    if after != target:
        raise ValueError("runtime installation changed during execution")
    record = replace(
        record,
        resolved={**record.resolved, "runtime_environment": observed},
        observed={**record.observed, "runtime_environment": observed},
        provenance={
            **record.provenance,
            "runtime_environment_observation": "before_controlled_cpu_binding",
        },
    )
    write_private(folder / "record.json", run_record_to_data(record))
    return {"environment": observed, "record_digest": run_record_digest(record)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    path = args.request.resolve()
    request, request_digest = read_bound_packet(path, 16 * 1024 * 1024)
    common = {"schema", "mode", "nonce", "metria_sha256"}
    extra = {
        "target",
        "study_name",
        "run_id",
        "spec",
        "measurement_config",
        "cpu_binding",
    }
    expected = common if request.get("mode") == "inspect" else common | extra
    if (
        set(request) != expected
        or request.get("schema") != REQUEST_SCHEMA
        or request.get("mode") not in ("inspect", "run")
    ):
        raise ValueError("unsupported runtime worker request")
    if request["metria_sha256"] != source_digest():
        raise ValueError(
            "runtime worker does not contain the pinned Metria implementation"
        )
    result = execute_request(request, path.parent)
    write_private(
        path.with_name("receipt.json"),
        {
            "schema": RECEIPT_SCHEMA,
            "nonce": request["nonce"],
            "request_sha256": request_digest,
            **result,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
