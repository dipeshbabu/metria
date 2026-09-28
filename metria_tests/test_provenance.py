from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from metria import ArtifactManifest
from metria.provenance import (
    artifact_manifest_from_data,
    artifact_manifest_to_data,
    artifact_manifest_to_json,
    validate_headline_manifest,
)


def _manifest():
    path = (
        Path(__file__).parents[1]
        / "artifacts/qualification/llamacpp-cpu-threads/headline-manifest.json"
    )
    return artifact_manifest_from_data(json.loads(path.read_text(encoding="utf-8")))


def test_shared_manifest_round_trip_preserves_source_and_license_unknowns():
    manifest = _manifest()
    validate_headline_manifest(manifest)
    encoded = artifact_manifest_to_json(manifest)
    assert (
        artifact_manifest_to_json(artifact_manifest_from_data(json.loads(encoded)))
        == encoded
    )
    assert "unknown in the historical qualification metadata" in encoded
    assert isinstance(
        artifact_manifest_from_data(json.loads(encoded)), ArtifactManifest
    )


@pytest.mark.parametrize(
    "field",
    [
        "code",
        "runtime",
        "model",
        "workload",
        "recipe_sha256",
        "hardware",
        "environment",
        "upstream",
    ],
)
def test_headline_provenance_cannot_omit_required_evidence(field):
    manifest = _manifest()
    source = dict(manifest.source)
    del source[field]
    with pytest.raises(ValueError, match="missing"):
        validate_headline_manifest(replace(manifest, source=source))


@pytest.mark.parametrize("field", ["code", "runtime", "model", "workload"])
def test_mutable_identity_is_not_headline_provenance(field):
    manifest = _manifest()
    source = {**manifest.source, field: {"identifier": "example", "revision": "main"}}
    with pytest.raises(ValueError, match="immutable"):
        validate_headline_manifest(replace(manifest, source=source))


@pytest.mark.parametrize("change", ["schema", "extra", "artifact_extra"])
def test_manifest_parser_rejects_unknown_versions_and_fields(change):
    data = artifact_manifest_to_data(_manifest())
    if change == "schema":
        data["schema"] = "metria.artifact_manifest.v999"
    elif change == "extra":
        data["unrecognized"] = True
    else:
        data["artifact"]["unrecognized"] = True
    with pytest.raises(ValueError):
        artifact_manifest_from_data(data)


@pytest.mark.parametrize(
    "metadata",
    [
        {"created_at": "2026-09-28T00:00:00"},
        {"created_at": "invalid"},
        {"license_status": ""},
        {"claim_scope": ""},
    ],
)
def test_headline_date_and_rights_are_explicit(metadata):
    manifest = _manifest()
    with pytest.raises(ValueError):
        validate_headline_manifest(
            replace(manifest, metadata={**manifest.metadata, **metadata})
        )


def test_repository_checker_rejects_missing_licenses_and_tampered_artifacts(tmp_path):
    import runpy
    import shutil

    root = Path(__file__).parents[1]
    checker = runpy.run_path(str(root / "tools/maintenance/check_provenance.py"))[
        "check_repository"
    ]
    for directory in ("", "components/kv-fidelity", "components/turboquant-reference"):
        for filename in ("LICENSE", "NOTICE"):
            destination = tmp_path / directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / directory / filename, destination)
    for relative in (
        "artifacts/headline-manifests.json",
        "artifacts/qualification/llamacpp-cpu-threads/headline-manifest.json",
        "artifacts/qualification/llamacpp-cpu-threads/threads-1-to-2/report.md",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    # The fixture deliberately contains one artifact, independent of additions
    # to the live repository's headline catalog.
    (tmp_path / "artifacts/headline-manifests.json").write_text(
        json.dumps(
            {
                "schema": "metria.headline_index.v1",
                "manifests": [
                    "artifacts/qualification/llamacpp-cpu-threads/headline-manifest.json"
                ],
            }
        ),
        encoding="utf-8",
    )
    checker(tmp_path)
    (tmp_path / "NOTICE").unlink()
    with pytest.raises(ValueError, match="licensing file"):
        checker(tmp_path)
    shutil.copyfile(root / "NOTICE", tmp_path / "NOTICE")
    (
        tmp_path
        / "artifacts/qualification/llamacpp-cpu-threads/threads-1-to-2/report.md"
    ).write_text("tampered")
    with pytest.raises(RuntimeError, match="SHA-256|size"):
        checker(tmp_path)
