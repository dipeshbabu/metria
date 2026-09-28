from __future__ import annotations

import pytest

from metria.integrations import turboquant_diagnostics as diagnostics
from metria.reporting import render_verification
from metria.verification import render_verification as legacy_render


def test_verification_renderer_keeps_its_compatibility_import():
    assert render_verification is legacy_render


@pytest.mark.parametrize(
    "system,release,machine,status",
    [
        ("Darwin", "24", "arm64", "supported"),
        ("Darwin", "24", "x86_64", "unsupported"),
        ("Linux", "6.8", "x86_64", "supported"),
        ("Linux", "microsoft-wsl2", "x86_64", "degraded"),
        ("Windows", "11", "AMD64", "unsupported"),
        ("Other", "1", "other", "unsupported"),
    ],
)
def test_library_diagnostic_policy_does_not_invent_accelerator_evidence(
    system, release, machine, status
):
    support = diagnostics.get_platform_support(
        system=system, release=release, machine=machine, environ={}
    )
    assert support["status"] == status
    fingerprint = diagnostics.platform_support_fingerprint(support)
    assert fingerprint.platform["system"] == system
    assert not fingerprint.accelerators
    assert (
        fingerprint.metadata["accelerator_detection"] == "runtime_or_adapter_required"
    )


def test_library_parsers_preserve_native_benchmark_and_perplexity_output():
    row = "| model | 1 GiB | 1 B | CPU | 2 | q8_0 | q8_0 | 1 | tg128 @ d4096 | 70.88 ± 1.27 |"
    assert diagnostics.parse_bench_tps(row) == [
        {"mode": "decode", "depth": 4096, "tps": 70.88, "stddev": 1.27, "ctk": "q8_0"}
    ]
    assert diagnostics.parse_ppl_final("Final estimate: PPL = 6.2109 +/- 0.33250") == (
        6.2109,
        0.3325,
    )
