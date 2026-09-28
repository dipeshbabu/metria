from __future__ import annotations

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "text",
    [
        "/Users/alice/models/model.gguf",
        "/home/bob/build",
        r"C:\Users\alice\model",
        "C:/Users/alice/model",
        "/root/scripts/run.sh",
    ],
)
def test_contributor_home_paths_are_rejected(text):
    tool = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/maintenance/check_local_paths.py")
    )
    assert tool["local_path_lines"]("heading\n" + text) == [2]


@pytest.mark.parametrize(
    "text",
    [
        "${HOME}/models/model.gguf",
        "/path/to/model.gguf",
        "components/kv-fidelity",
        "/home/USER/cache",
        "C:/Users/<user>/model",
    ],
)
def test_portable_paths_and_placeholders_are_allowed(text):
    tool = runpy.run_path(
        str(Path(__file__).parents[1] / "tools/maintenance/check_local_paths.py")
    )
    assert tool["local_path_lines"](text) == []
