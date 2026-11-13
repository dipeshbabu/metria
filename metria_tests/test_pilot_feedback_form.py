"""Participant feedback issues convert to validated pilot notes without inference."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pilot_notes_from_issue", ROOT / "tools/maintenance/pilot_notes_from_issue.py"
)
assert _spec and _spec.loader
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

BODY = """### Permission

- [X] I am a pilot participant (not a Metria maintainer) and I give permission for this feedback to be recorded and published with the pilot evidence.

### How should we credit you?

Inference team A

### Inference change you verified

vLLM FP8 KV cache

### Metria version and recipe digest

metria 0.1.6; recipe 34a99074

### Your keep/reject decision

reject

### Why did you decide that?

Candidate task checks dropped below our threshold.

### Your acceptance criteria

Candidate pass rate must match the reference.

### Setup time in minutes

25

### Setup friction

- WSL needed a restart
- Wheel download was slow

### Would you use Metria again for this kind of change?

yes

### Retained evidence (optional)

_No response_
"""
URL = "https://github.com/dipeshbabu/metria/issues/999"


def test_feedback_issue_becomes_participant_notes() -> None:
    notes = tool.notes_from_issue(BODY, URL)
    assert notes["kind"] == "participant_feedback"
    assert notes["decision"] == "reject"
    assert notes["setup_seconds"] == 1500
    assert notes["friction"] == ["WSL needed a restart", "Wheel download was slow"]
    assert notes["would_reuse"] is True
    assert notes["permission_to_share"] is True
    assert notes["feedback_source"] == URL


def test_missing_permission_records_nothing() -> None:
    body = BODY.replace("- [X] I am", "- [ ] I am")
    with pytest.raises(ValueError, match="permission"):
        tool.notes_from_issue(body, URL)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("\n25\n", "\nabout half an hour\n", "minutes"),
        ("\nreject\n", "\nmaybe\n", "decision"),
    ],
)
def test_malformed_answers_are_rejected(old, new, match) -> None:
    with pytest.raises(ValueError, match=match):
        tool.notes_from_issue(BODY.replace(old, new, 1), URL)


def test_unsure_reuse_stays_null_and_form_matches_tool() -> None:
    body = BODY.replace("\nyes\n", "\nnot sure\n", 1)
    assert tool.notes_from_issue(body, URL)["would_reuse"] is None
    form = (ROOT / ".github/ISSUE_TEMPLATE/pilot_feedback.yml").read_text()
    for heading in tool._sections(BODY):
        assert heading in form
