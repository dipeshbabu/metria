"""Reusable policy/parsers for the historical TurboQuant diagnostic integration.

Declared platform eligibility is not observed accelerator evidence.
"""

from __future__ import annotations

import os
import platform
import re
from typing import Optional

from ..identity import HardwareFingerprint

PLATFORM_SUPPORT_SCHEMA = "turbo_diag.platform_support.v1"
ALL_DIAGNOSTIC_SECTIONS = tuple(range(1, 14))


def get_platform_support(
    *,
    system: Optional[str] = None,
    release: Optional[str] = None,
    machine: Optional[str] = None,
    environ: Optional[dict[str, str]] = None,
) -> dict:
    """Return the declared support contract for a host environment.

    Optional inputs make the policy independently testable without impersonating
    platform modules. The returned dictionary is embedded in diagnostic JSON.
    """
    detected_system = system if system is not None else platform.system()
    detected_release = release if release is not None else platform.release()
    detected_machine = machine if machine is not None else platform.machine()
    detected_environ = os.environ if environ is None else environ
    release_lower = detected_release.lower()
    machine_lower = detected_machine.lower()
    is_wsl = detected_system == "Linux" and (
        "microsoft" in release_lower
        or "wsl" in release_lower
        or bool(detected_environ.get("WSL_INTEROP"))
        or bool(detected_environ.get("WSL_DISTRO_NAME"))
    )

    def contract(
        *,
        environment: str,
        status: str,
        reason: str,
        action: str,
        monitoring: str,
        supported_backends: list[str],
        degraded_backends: list[str],
        unavailable_backends: list[str],
        supported_sections: list[int],
        degraded_sections: list[int],
        unavailable_sections: list[int],
    ) -> dict:
        return {
            "schema": PLATFORM_SUPPORT_SCHEMA,
            "platform": detected_system,
            "release": detected_release,
            "machine": detected_machine,
            "environment": environment,
            "status": status,
            "reason": reason,
            "action": action,
            "monitoring": monitoring,
            "backend_probe": {
                "status": "not-run",
                "returncode": None,
            },
            "gpu_backends": {
                "supported": supported_backends,
                "degraded": degraded_backends,
                "unavailable": unavailable_backends,
            },
            "sections": {
                "supported": supported_sections,
                "degraded": degraded_sections,
                "skipped": [],
                "unavailable": unavailable_sections,
            },
        }

    all_sections = list(ALL_DIAGNOSTIC_SECTIONS)
    if detected_system == "Darwin" and machine_lower in {"arm64", "aarch64"}:
        return contract(
            environment="macos-apple-silicon",
            status="supported",
            reason="Apple Silicon hardware and Metal telemetry are supported.",
            action="Run the Bash launcher or the Python diagnostic directly.",
            monitoring="supported",
            supported_backends=["metal"],
            degraded_backends=[],
            unavailable_backends=["cuda", "rocm", "vulkan", "cpu"],
            supported_sections=all_sections,
            degraded_sections=[],
            unavailable_sections=[],
        )
    if detected_system == "Darwin":
        return contract(
            environment="macos-intel",
            status="unsupported",
            reason="The macOS diagnostic contract covers Apple Silicon only.",
            action="Use a supported Apple Silicon or native Linux host.",
            monitoring="unavailable",
            supported_backends=[],
            degraded_backends=[],
            unavailable_backends=["metal", "cuda", "rocm", "vulkan", "cpu"],
            supported_sections=[],
            degraded_sections=[],
            unavailable_sections=all_sections,
        )
    if is_wsl:
        return contract(
            environment="wsl",
            status="degraded",
            reason=(
                "WSL can run Linux llama.cpp binaries, but host thermal, power, "
                "storage, and GPU-topology telemetry is incomplete."
            ),
            action=(
                "Use a WSL CUDA build and treat hardware/load sections as "
                "partial evidence."
            ),
            monitoring="degraded",
            supported_backends=["cuda"],
            degraded_backends=[],
            unavailable_backends=["metal", "rocm", "vulkan", "cpu"],
            supported_sections=[3, 5, 6, 7, 8, 9, 10, 11, 13],
            degraded_sections=[1, 2, 4, 12],
            unavailable_sections=[],
        )
    if detected_system == "Linux":
        return contract(
            environment="linux",
            status="supported",
            reason="Native Linux diagnostic and monitoring probes are supported.",
            action="Use a matching Linux llama.cpp build for the selected backend.",
            monitoring="supported",
            supported_backends=["cuda"],
            degraded_backends=["rocm", "vulkan", "cpu"],
            unavailable_backends=["metal"],
            supported_sections=all_sections,
            degraded_sections=[],
            unavailable_sections=[],
        )
    if detected_system == "Windows":
        return contract(
            environment="windows-native",
            status="unsupported",
            reason=(
                "Native Windows lacks the Linux/macOS telemetry contract and "
                "the supported Bash launcher."
            ),
            action=(
                "Run the diagnostic inside WSL with Linux llama.cpp binaries, "
                "or use a native Linux/macOS host."
            ),
            monitoring="unavailable",
            supported_backends=[],
            degraded_backends=[],
            unavailable_backends=["metal", "cuda", "rocm", "vulkan", "cpu"],
            supported_sections=[],
            degraded_sections=[],
            unavailable_sections=all_sections,
        )
    return contract(
        environment="unknown",
        status="unsupported",
        reason=f"No diagnostic support contract exists for {detected_system!r}.",
        action="Use a supported Apple Silicon/macOS or native Linux host.",
        monitoring="unavailable",
        supported_backends=[],
        degraded_backends=[],
        unavailable_backends=["metal", "cuda", "rocm", "vulkan", "cpu"],
        supported_sections=[],
        degraded_sections=[],
        unavailable_sections=all_sections,
    )


def parse_bench_tps(output: str) -> list[dict]:
    """Parse llama-bench table output for tok/s values.

    Returns list of dicts with keys: mode, depth, tps, stddev, ctk
    """
    results: list[dict] = []
    for line in output.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 10:
            continue

        # Find test column (ppXXXX or tgXXXX)
        test_col = ""
        tps_col = ""
        for i, col in enumerate(cols):
            if re.match(r"(pp|tg)\d+", col):
                test_col = col
                if i + 1 < len(cols):
                    tps_col = cols[i + 1]
                break

        if not test_col:
            continue

        # Mode + depth
        mode = ""
        depth = 0
        if test_col.startswith("pp") and "+tg" in test_col:
            mode = "combined"
            m = re.match(r"pp(\d+)\+tg(\d+)", test_col)
            if m:
                depth = int(m.group(1))
        elif test_col.startswith("pp"):
            mode = "prefill"
            m = re.match(r"pp(\d+)", test_col)
            if m:
                depth = int(m.group(1))
        elif test_col.startswith("tg"):
            mode = "decode"
            m = re.search(r"d(\d+)", test_col)
            if m:
                depth = int(m.group(1))

        # tok/s
        tps = 0.0
        stddev = 0.0
        m = re.match(r"([\d.]+)\s*\u00b1\s*([\d.]+)", tps_col)
        if m:
            tps = float(m.group(1))
            stddev = float(m.group(2))
        else:
            m = re.match(r"[\d.]+", tps_col)
            if m:
                tps = float(m.group())

        # Extract cache type from row
        row_ctk = ""
        for col in cols:
            cs = col.strip()
            if cs in ("q8_0", "turbo3", "turbo4", "f16", "q4_0"):
                row_ctk = cs
                break

        results.append(
            {
                "mode": mode,
                "depth": depth,
                "tps": tps,
                "stddev": stddev,
                "ctk": row_ctk,
            }
        )

    return results


def parse_ppl_final(output: str) -> tuple[float, float]:
    """Extract final PPL estimate from perplexity output.

    Returns (ppl, stddev) or (0, 0) if not found.
    """
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", output)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def platform_support_fingerprint(support: dict) -> HardwareFingerprint:
    """Map supplied platform facts without inventing accelerator presence."""
    return HardwareFingerprint(
        platform={
            "system": support["platform"],
            "release": support["release"],
            "machine": support["machine"],
        },
        metadata={
            "diagnostic_policy_schema": support["schema"],
            "diagnostic_environment": support["environment"],
            "diagnostic_status": support["status"],
            "accelerator_detection": "runtime_or_adapter_required",
        },
    )
