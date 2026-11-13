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


def test_legacy_rendering_reproduces_every_retained_report_exactly() -> None:
    root = MANIFEST.parents[3]
    reports = sorted(root.glob("qualification/**/report.md"))
    assert reports
    for report in reports:
        manifest_path = report.with_name("verification.json")
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        text = report.read_text(encoding="utf-8").replace("\r\n", "\n")
        assert text in {
            render_verification(manifest),
            render_verification(manifest, legacy_identifiers=True),
        }, report
