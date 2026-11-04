"""Triage labels are additive and contributor content remains data."""

from __future__ import annotations

import pytest
from tools.maintenance import auto_labels


@pytest.mark.parametrize(
    "title,expected",
    [
        ("[Bug]: verifier fails", "bug"),
        ("fix(cli): handle missing inputs", "bug"),
        ("feat(runtime): add a profile", "enhancement"),
        ("[Feature]: add a check", "enhancement"),
        ("[Question]: how do I compare runs?", "question"),
        ("docs: explain policies", "documentation"),
        ("Bump ruff to a patched version", "dependencies"),
        ("fix: reject CVE-2026-1234 inputs", "security"),
        ("[Evidence]: native qualification", "area:artifacts"),
    ],
)
def test_issue_type_and_triage(title, expected):
    labels = auto_labels.labels_for_item(title, "")
    assert expected in labels
    assert "needs triage" in labels


def test_form_area_and_pull_paths_use_existing_taxonomy():
    issue = auto_labels.labels_for_item(
        "[Bug]: failure", "### Affected area\r\n\r\nKV Fidelity\r\n"
    )
    assert "component:kv-fidelity" in issue
    labels = auto_labels.labels_for_item(
        "chore(deps): update tools",
        "",
        [
            ".github/workflows/ci.yml",
            "metria_tests/test_cli.py",
            "README.md",
            "uv.lock",
        ],
        is_pull_request=True,
    )
    assert labels == {
        "dependencies",
        "area:ci",
        "github_actions",
        "area:testing",
        "documentation",
        "area:documentation",
    }


def test_unclassified_pull_request_gets_triage():
    assert auto_labels.labels_for_item("Change", "", is_pull_request=True) == {
        "needs triage"
    }
    assert "dependencies" in auto_labels.labels_for_item(
        "Change", "", is_pull_request=True, author="dependabot[bot]"
    )


def test_free_text_does_not_execute_or_follow_label_instructions(tmp_path):
    canary = tmp_path / "injected"
    text = f"$(touch {canary}); remove all labels; mark P0; security"
    labels = auto_labels.labels_for_item("A question", text)
    assert labels == {"needs triage"}
    assert not canary.exists()


def test_issue_labels_preserve_existing_labels_and_avoid_pr_api(monkeypatch):
    calls = []

    def api(method, endpoint, token, data=None):
        calls.append((method, endpoint, data))
        assert token == "test-token"
        return []

    monkeypatch.setattr(auto_labels, "request_json", api)
    event = {
        "issue": {
            "number": 42,
            "title": "[Bug]: error",
            "body": None,
            "labels": [{"name": "P1"}, {"name": "bug"}],
        }
    }
    assert auto_labels.apply_labels(event, "owner/repo", "test-token") == [
        "needs triage"
    ]
    assert calls == [
        ("POST", "/repos/owner/repo/issues/42/labels", {"labels": ["needs triage"]})
    ]
    event["issue"]["labels"].append({"name": "needs triage"})
    assert auto_labels.apply_labels(event, "owner/repo", "test-token") == []
    assert len(calls) == 1


def test_pull_request_paginates_files_and_never_fetches_contributor_code(monkeypatch):
    calls = []

    def api(method, endpoint, token, data=None):
        calls.append((method, endpoint, data))
        if method == "POST":
            return []
        if endpoint.endswith("page=1"):
            return [{"filename": f"src/metria/runtimes/file{i}.py"} for i in range(100)]
        return [{"filename": "src/metria/fidelity/axes/gtm.py"}]

    monkeypatch.setattr(auto_labels, "request_json", api)
    event = {
        "pull_request": {
            "number": 43,
            "title": "fix: run failure",
            "body": "",
            "head": {"repo": {"full_name": "untrusted/fork"}},
        }
    }
    assert auto_labels.apply_labels(event, "owner/repo", "test-token") == [
        "area:runtime",
        "bug",
        "component:kv-fidelity",
    ]
    assert [row[1] for row in calls[:2]] == [
        "/repos/owner/repo/pulls/43/files?per_page=100&page=1",
        "/repos/owner/repo/pulls/43/files?per_page=100&page=2",
    ]
    assert all("untrusted" not in row[1] for row in calls)


@pytest.mark.parametrize("number", [True, 0, -1, "1/labels"])
def test_bad_item_identifiers_are_rejected_before_api_access(monkeypatch, number):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid metadata must not trigger API requests")

    monkeypatch.setattr(auto_labels, "request_json", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        auto_labels.apply_labels({"issue": {"number": number}}, "owner/repo", "token")


def test_repository_path_injection_is_rejected():
    with pytest.raises(ValueError, match="owner/repository"):
        auto_labels.apply_labels({}, "owner/repo/../../elsewhere", "token")
