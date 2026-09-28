"""Check package license files and explicitly designated headline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from metria.artifacts import verify_artifact
from metria.provenance import artifact_manifest_from_data, validate_headline_manifest


def check_repository(root: Path) -> None:
    root = root.resolve()
    for package in (
        root,
        root / "components/kv-fidelity",
        root / "components/turboquant-reference",
    ):
        for filename in ("LICENSE", "NOTICE"):
            path = package / filename
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                raise ValueError(
                    f"missing package licensing file: {path.relative_to(root)}"
                )
    index = json.loads(
        (root / "artifacts/headline-manifests.json").read_text(encoding="utf-8")
    )
    if index.get("schema") != "metria.headline_index.v1" or not isinstance(
        index.get("manifests"), list
    ):
        raise ValueError("invalid headline artifact index")
    for relative in index["manifests"]:
        manifest_path = (root / relative).resolve()
        if not manifest_path.is_relative_to(root):
            raise ValueError("headline manifest escapes repository")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = artifact_manifest_from_data(data)
        validate_headline_manifest(manifest)
        if not isinstance(manifest.path, str) or Path(manifest.path).is_absolute():
            raise ValueError("headline artifact requires a repository-relative path")
        artifact_path = (root / manifest.path).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError("headline artifact escapes repository")
        verify_artifact(
            manifest, artifact_path, max_bytes=max(1, manifest.size_bytes or 0)
        )


def main() -> int:
    check_repository(Path(__file__).resolve().parents[2])
    print("Package licensing files and headline artifact provenance validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
