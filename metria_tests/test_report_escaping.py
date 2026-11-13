"""Markdown reports keep identifiers readable and never emit live HTML."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from metria.reporting import render_verification

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "artifacts/qualification/fp8-kv-cache-verify-l4/verification/verification.json"
)


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_identifiers_render_with_plain_quotes() -> None:
    report = render_verification(_manifest())
    assert '      Category "classification": 0/4 diverged' in report
    assert 'Failed "sentiment" on "0:positive" (exact_text)' in report
    assert "&quot;" not in report


def test_hostile_identifiers_are_still_escaped() -> None:
    manifest = copy.deepcopy(_manifest())
    for analysis in manifest["analyses"]:
        divergence = analysis["diagnostics"].get("divergence")
        if divergence:
            divergence["by_category"][0]["category"] = "<script>alert(1)</script>"
    report = render_verification(manifest)
    assert "<script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
