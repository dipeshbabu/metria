"""Distribution metadata must preserve KV Fidelity's import and CLI contracts."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

from kv_fidelity import __version__


def test_distribution_identity_matches_runtime_contract() -> None:
    manifest = Path(__file__).parents[1] / "pyproject.toml"
    with manifest.open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["name"] == "kv-fidelity"
    assert project["version"] == __version__
    assert project["scripts"]["kv-fidelity"] == "kv_fidelity.cli:main"
