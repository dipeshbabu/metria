"""Engine-independent result and capability contracts for KV Fidelity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ModelSpec = str | Path


class BackendCapabilityError(RuntimeError):
    """Raised when a backend doesn't support a feature that an axis requires.

    Backends should raise this with a clear remediation hint so the user
    knows whether to switch backends or skip the axis.
    """


@dataclass
class CompletionResult:
    """Result of a backend.run_completion call."""

    text: str  # post-noise-strip completion text
    n_tokens: int  # tokens actually decoded
    metadata: dict = field(default_factory=dict)  # backend-specific extras


@dataclass
class TrajectoryResult:
    """Result of a backend.run_completion_trajectory call."""

    token_ids: list[int]  # actual sampled IDs at decode time
    metadata: dict = field(default_factory=dict)


@dataclass
class KLDResult:
    """Result of a backend.run_kld call (Axis B)."""

    mean_kld: float  # nats
    ppl: Optional[float] = None
    rms_dp_pct: Optional[float] = None
    same_topp_pct: Optional[float] = None
    chunks: int = 0
    ctx: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _TopKKLDMetrics:
    """Validated aggregate metrics for paired top-k backend responses."""

    mean_kld: float
    rms_dp_pct: Optional[float]
    same_topp_pct: float
    n_positions_total: int
    n_positions_scored: int
    n_positions_skipped: int
