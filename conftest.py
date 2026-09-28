"""Required evidence-fixture inventory and explicit CI skip reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_SKIPPED = []


def pytest_addoption(parser):
    parser.addoption(
        "--skip-report", default=None, help="write skipped test IDs/reasons as JSON"
    )


def pytest_sessionstart(session):
    _SKIPPED.clear()
    root = Path(__file__).parent
    required = [
        "tools/qualification/vllm-smollm2-135m.json",
        "artifacts/qualification/llamacpp-cpu-threads/threads-1-to-2/reference.run.json",
        "artifacts/qualification/llamacpp-cpu-threads/threads-1-to-2/candidate.run.json",
        "artifacts/qualification/vllm-0.30.0/cpu/qualification.run.json",
        "artifacts/qualification/vllm-0.30.0/cuda/qualification.run.json",
        "src/metria/fidelity/prompts/v0.1.jsonl",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise pytest.UsageError(
            "required retained evidence fixtures are missing: " + ", ".join(missing)
        )


def pytest_runtest_logreport(report):
    if report.skipped:
        _SKIPPED.append({"nodeid": report.nodeid, "reason": str(report.longrepr)})


def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--skip-report")
    if path:
        Path(path).write_text(
            json.dumps({"skipped": _SKIPPED}, indent=2) + "\n", encoding="utf-8"
        )
