"""Apply additive, deterministic labels without executing contributor content."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

API = "https://api.github.com"
AREA_FIELDS = {
    "metria core or cli": {"area:cli", "area:verification"},
    "kv fidelity": {"component:kv-fidelity"},
    "turboquant reference": {"area:analysis"},
    "diagnostics": {"area:capabilities"},
    "validation or benchmark tools": {"area:benchmarking"},
    "packaging, ci, or release automation": {"area:ci", "area:release"},
    "documentation or research record": {"documentation", "area:documentation"},
    "documentation or research": {"documentation", "area:documentation"},
    "development or packaging": {"area:ci"},
}
PATH_AREAS = {
    ".github/": "area:ci",
    "tools/ci/": "area:ci",
    "tools/maintenance/": "area:ci",
    "tools/qualification/": "area:qualification",
    "src/metria/runtimes/": "area:runtime",
    "src/metria/measurements/": "area:measurement",
    "src/metria/verification": "area:verification",
    "src/metria/preparation": "area:verification",
    "src/metria/policies": "area:policy",
    "src/metria/identity": "area:identity",
    "src/metria/comparison": "area:comparison",
    "src/metria/reporting": "area:reporting",
    "examples/": "area:examples",
    "artifacts/": "area:artifacts",
    "benchmarks/": "area:benchmarking",
}


def _title_labels(title: str, author: str) -> set[str]:
    title = title.casefold().strip()
    labels = set()
    patterns = {
        "bug": r"^(?:\[bug\]|fix(?:\([^)]*\))?!?:)",
        "enhancement": r"^(?:\[feature\]|feat(?:\([^)]*\))?!?:)",
        "question": r"^\[question\]",
        "documentation": r"^(?:\[docs?\]|docs(?:\([^)]*\))?:)",
        "dependencies": r"^(?:bump\b|(?:chore|build)\(deps(?:-dev)?\):)",
        "area:artifacts": r"^\[evidence\]",
        "security": r"\b(?:security|vulnerabilit\w*|cve-\d|ghsa-)",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, title):
            labels.add(label)
    if author == "dependabot[bot]":
        labels.add("dependencies")
    return labels


def _path_labels(filename: str) -> set[str]:
    labels = {
        label for prefix, label in PATH_AREAS.items() if filename.startswith(prefix)
    }
    parts = filename.split("/")
    basename = parts[-1]
    if filename.startswith(("docs/", "research/")) or basename.endswith(".md"):
        labels.update(("documentation", "area:documentation"))
    if filename.startswith(("metria_tests/", "tests/")) or "tests" in parts:
        labels.add("area:testing")
    if filename.startswith(("components/kv-fidelity/", "src/metria/fidelity/")):
        labels.add("component:kv-fidelity")
    if filename.startswith(".github/workflows/"):
        labels.add("github_actions")
    if basename in {"pyproject.toml", "uv.lock", ".pre-commit-config.yaml"}:
        labels.add("dependencies")
    if filename == ".pre-commit-config.yaml":
        labels.add("pre_commit")
    if basename in {"LICENSE", "NOTICE", "CITATION.cff"}:
        labels.add("area:provenance")
    if filename == "SECURITY.md":
        labels.add("security")
    if filename in {"src/metria/cli.py", "src/metria/onboarding.py"}:
        labels.add("area:cli")
    if filename.startswith("docs/releases/"):
        labels.add("area:release")
    return labels


def labels_for_item(
    title: str,
    body: str,
    filenames: Sequence[str] = (),
    *,
    is_pull_request: bool = False,
    author: str = "",
) -> set[str]:
    labels = _title_labels(title, author)
    for filename in filenames:
        labels.update(_path_labels(filename))
    fields = re.findall(
        r"(?m)^### (?:Affected area|Target area|Area)\s*\r?\n+([^\r\n]+)", body
    )
    for value in fields:
        labels.update(AREA_FIELDS.get(value.strip().casefold(), set()))
    if "documentation" in labels:
        labels.add("area:documentation")
    if not is_pull_request or not labels:
        labels.add("needs triage")
    return labels


def request_json(method: str, endpoint: str, token: str, data: Any = None) -> Any:
    request = Request(
        API + endpoint,
        data=None if data is None else json.dumps(data, allow_nan=False).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def pull_files(repository: str, number: int, token: str) -> list[str]:
    filenames: list[str] = []
    for page in range(1, 31):
        rows = request_json(
            "GET",
            f"/repos/{repository}/pulls/{number}/files?per_page=100&page={page}",
            token,
        )
        filenames.extend(row["filename"] for row in rows)
        if len(rows) < 100:
            break
    return filenames


def apply_labels(event: Mapping[str, Any], repository: str, token: str) -> list[str]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ValueError("expected the trusted owner/repository environment value")
    is_pr = "pull_request" in event
    item = event["pull_request" if is_pr else "issue"]
    number = item["number"]
    if type(number) is not int or number <= 0:
        raise ValueError("GitHub item number must be a positive integer")
    filenames = pull_files(repository, number, token) if is_pr else []
    desired = labels_for_item(
        item["title"],
        item.get("body") or "",
        filenames,
        is_pull_request=is_pr,
        author=item.get("user", {}).get("login", ""),
    )
    existing = {label["name"] for label in item.get("labels", [])}
    added = sorted(desired - existing)
    if added:
        request_json(
            "POST",
            f"/repos/{repository}/issues/{number}/labels",
            token,
            {"labels": added},
        )
    return added


def main() -> int:
    path = Path(os.environ["GITHUB_EVENT_PATH"])
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("GitHub event payload exceeds the input bound")
    event = json.loads(path.read_text(encoding="utf-8"))
    labels = apply_labels(
        event, os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"]
    )
    print("Added labels: " + (", ".join(labels) if labels else "none needed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
