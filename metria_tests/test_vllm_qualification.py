from __future__ import annotations

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
