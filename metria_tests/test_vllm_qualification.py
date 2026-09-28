from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from metria.protocols import InferenceBatch


def _tool():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "tools/qualification/qualify_vllm.py")
    )


class Session:
    def __init__(self, change=False):
        self.resets = 0
        self.change = change

    def infer(self, requests, capture=()):
        token = 2 if self.change and self.resets else 1
        return InferenceBatch(outputs=("unused",), captures={"token_ids": ((token,),)})

    def reset(self, scope="measurement"):
        assert scope == "measurement"
        self.resets += 1


def test_real_qualification_exercises_reset_and_keeps_capture_contract():
    protocol = _tool()["ResetCapture"]()
    session = Session()
    result = protocol.execute(
        session, {}, {"prompts": [{"id": "one", "prompt": "fixture"}]}
    )
    assert session.resets == 1
    assert result.evidence["qualification_reset"]["repeated_trajectories_match"]
    assert result.evidence["schema"] == "metria.trajectory_capture.v1"


def test_qualification_rejects_post_reset_drift():
    protocol = _tool()["ResetCapture"]()
    with pytest.raises(RuntimeError, match="changed after reset"):
        protocol.execute(
            Session(change=True), {}, {"prompts": [{"id": "one", "prompt": "fixture"}]}
        )


def test_qualification_path_redaction_is_recursive_and_preserves_values():
    redact = _tool()["_redact_paths"]
    data = {
        "model": ["/private/cache/model", {"path": "/private/cache/model/file"}],
        "token": 3,
    }
    assert redact(data, {"/private/cache/model": "${MODEL_SNAPSHOT}"}) == {
        "model": ["${MODEL_SNAPSHOT}", {"path": "${MODEL_SNAPSHOT}/file"}],
        "token": 3,
    }


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_retained_real_qualification_hashes_capture_and_reset(backend):
    from metria.records import load_run_record

    root = Path(__file__).parents[1]
    directory = root / "artifacts/qualification/vllm-0.30.0" / backend
    summary = json.loads((directory / "qualification.json").read_text(encoding="utf-8"))
    record_path = directory / summary["record"]["path"]
    assert (
        hashlib.sha256(record_path.read_bytes()).hexdigest()
        == summary["record"]["sha256"]
    )
    record = load_run_record(record_path)
    assert record.status.value == summary["status"] == "completed"
    capture = record.evidence["measurements"]["kv_fidelity.decode_time_trajectory"]
    assert capture["qualification_reset"]["repeated_trajectories_match"] is True
    assert record.observed["reset_count"] == 1
    assert len(capture["prompts"]) == 2
    assert all(len(prompt["token_ids"]) == 8 for prompt in capture["prompts"])
    assert record.observed["identity"]["runtime"]["status"] == "verified"
    assert "/root/" not in record_path.read_text(encoding="utf-8")
    if backend == "cuda":
        gpu = summary["qualification"]["hardware"]["accelerators"][0]
        assert gpu["name"] == "NVIDIA GeForce GTX 1650"
        assert gpu["compute_capability"] == [7, 5]
        assert gpu["driver_version"] == "566.36"
