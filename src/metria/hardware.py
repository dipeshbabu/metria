"""Portable, privacy-conscious hardware fingerprint capture for Metria."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
from collections.abc import Mapping
from typing import Any

from .identity import HardwareFingerprint


def _hostname_hash() -> str:
    hostname = socket.gethostname()
    return hashlib.sha256(hostname.encode("utf-8")).hexdigest()


def capture_hardware_fingerprint(
    *,
    metadata: Mapping[str, Any] | None = None,
) -> HardwareFingerprint:
    """Capture stdlib-observable host/software identity without raw hostnames.

    Accelerator probing is intentionally not guessed from model/runtime names or
    environment variables. Runtimes may add authoritative accelerator evidence
    later through their observed state. The host name is retained only as a
    SHA-256 fingerprint so two records can identify the same host without
    publishing the machine name.
    """

    uname = platform.uname()
    return HardwareFingerprint(
        platform={
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        host={
            "hostname_sha256": _hostname_hash(),
            "cpu_count": os.cpu_count(),
        },
        accelerators=(),
        software={
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": os.path.basename(sys.executable),
        },
        metadata={
            "accelerator_detection": "runtime_or_adapter_required",
            **dict(metadata or {}),
        },
    )
