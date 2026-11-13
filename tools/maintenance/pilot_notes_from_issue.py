#!/usr/bin/env python3
"""Convert a submitted "Pilot feedback" issue into `metria pilot record` notes.

The issue form (``.github/ISSUE_TEMPLATE/pilot_feedback.yml``) collects exactly
the fields that ``metria.pilot.validate_notes`` requires. This tool copies the
participant's words without inference: it refuses an issue whose permission box
is unchecked and sets ``permission_to_share`` only from that box.

Usage::

    gh issue view 123 --json body,url > issue.json
    python tools/maintenance/pilot_notes_from_issue.py issue.json --output notes.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from metria.pilot import validate_notes

_HEADING = re.compile(r"^### (.+?)\s*$", re.MULTILINE)
_CONSENT = "I am a pilot participant (not a Metria maintainer)"
_NO_RESPONSE = "_No response_"


def _sections(body: str) -> dict[str, str]:
    """Split GitHub issue-form markdown into heading -> answer text."""

    matches = list(_HEADING.finditer(body))
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[match.end() : end].strip()
        sections[match.group(1).strip()] = "" if text == _NO_RESPONSE else text
    return sections


def notes_from_issue(body: str, url: str) -> dict[str, Any]:
    """Return validated participant notes for one submitted feedback issue."""

    answers = _sections(body)
    consent = answers.get("Permission", "")
    if f"- [x] {_CONSENT}" not in consent.replace("[X]", "[x]"):
        raise ValueError("the participant did not grant permission; nothing recorded")
    try:
        minutes = float(answers["Setup time in minutes"])
    except (KeyError, ValueError):
        raise ValueError("setup time must be a number of minutes") from None
    reuse = answers.get(
        "Would you use Metria again for this kind of change?", "not sure"
    )
    notes = {
        "kind": "participant_feedback",
        "reporter": answers.get("How should we credit you?", ""),
        "decision": answers.get("Your keep/reject decision", ""),
        "rationale": answers.get("Why did you decide that?", ""),
        "acceptance_criteria": answers.get("Your acceptance criteria", ""),
        "setup_seconds": minutes * 60,
        "friction": [
            line.strip("-* ").strip()
            for line in answers.get("Setup friction", "").splitlines()
            if line.strip("-* ").strip()
        ],
        "would_reuse": {"yes": True, "no": False}.get(reuse),
        "permission_to_share": True,
        "feedback_source": url,
    }
    return validate_notes(notes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "issue", type=Path, help="`gh issue view --json body,url` output"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    issue = json.loads(args.issue.read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    try:
        notes = notes_from_issue(issue["body"], issue["url"])
    except (KeyError, ValueError) as exc:
        print(f"pilot_notes_from_issue: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(
        json.dumps(notes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.output}; record it with `metria pilot record --notes {args.output}`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
