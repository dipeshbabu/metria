"""Abstract Backend interface for KV Fidelity.

A backend wraps an inference engine (llama.cpp, MLX, vLLM, …) and exposes
the four primitives KV Fidelity axes need:

  - ``run_completion``       text-in/text-out with chat template + KV config
  - ``run_completion_trajectory``  decode-time token-ID capture
  - ``run_kld``              per-token KL divergence vs a reference, on a corpus
  - ``tokenize_to_ids``      tokenization for edit-distance and unit-matching
  - ``detect_thinking_mode`` runtime probe so axes can adapt n_predict / pre-fill
  - ``model_metadata``       framework version stamp + any backend-specific notes
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

from ..contracts import (
    BackendCapabilityError as BackendCapabilityError,
)
from ..contracts import (
    CompletionResult as CompletionResult,
)
from ..contracts import (
    KLDResult as KLDResult,
)
from ..contracts import (
    ModelSpec as ModelSpec,
)
from ..contracts import (
    TrajectoryResult as TrajectoryResult,
)
from ..contracts import (
    _TopKKLDMetrics as _TopKKLDMetrics,
)
from ..measurement_math import (
    aggregate_topk_kld,
    full_token_chunks,
)
from ..measurement_math import (
    approximate_topk_kl as approximate_topk_kl,
)

_aggregate_topk_kld = aggregate_topk_kld
_full_token_chunks = full_token_chunks


class Backend(abc.ABC):
    """Abstract KV Fidelity backend.

    Implementations must be importable without their underlying inference
    engine being installed (use lazy imports inside methods). This lets a
    user with only llama.cpp run KV Fidelity without paying the cost of
    importing mlx or vllm at startup.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def run_completion(
        self,
        *,
        model: ModelSpec,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,
        n_gpu_layers: int = 99,
        seed: int = 42,
        temperature: float = 0.0,
        timeout: float = 300.0,
        apply_chat_template: bool = True,
        system: Optional[str] = None,
        reasoning: str = "off",
    ) -> CompletionResult: ...

    @abc.abstractmethod
    def run_completion_trajectory(
        self,
        *,
        model: ModelSpec,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,
        n_gpu_layers: int = 99,
        seed: int = 42,
        temperature: float = 0.0,
        timeout: float = 300.0,
        apply_chat_template: bool = True,
        system: Optional[str] = None,
    ) -> TrajectoryResult: ...

    @abc.abstractmethod
    def run_kld(
        self,
        *,
        model: ModelSpec,
        corpus: Path,
        ref_kv_str: str,
        cand_kv_str: str,
        chunks: int = 32,
        ctx: int = 512,
        n_gpu_layers: int = 99,
    ) -> KLDResult: ...

    @abc.abstractmethod
    def tokenize_to_ids(
        self,
        *,
        model: ModelSpec,
        text: str,
        timeout: float = 120.0,
    ) -> list[int]: ...

    def detect_thinking_mode(
        self,
        *,
        model: ModelSpec,
        timeout: float = 30.0,
    ) -> tuple[bool, list[str]]:
        """Run a tiny probe and return ``(detected, markers_found)``.

        Default implementation issues a "What is 2+2?" generation and
        scans the response for canonical thinking markers. Subclasses can
        override with a cheaper signal (read GGUF chat_template, etc.).
        """
        markers = (
            "<think>",
            "</think>",
            "<|thinking|>",
            "<|end_thinking|>",
            "<|channel|>analysis",
            "<|channel|>commentary",
            "[Start thinking]",
            "[End thinking]",
            "<thinking>",
            "</thinking>",
        )
        try:
            result = self.run_completion(
                model=model,
                prompt="What is 2+2? Answer briefly.",
                kv_config_str="ctk=f16,ctv=f16",
                n_predict=64,
                ctx=128,
                temperature=0.0,
                seed=42,
                timeout=timeout,
            )
        except Exception:
            return False, []
        text = result.text or ""
        hit = [m for m in markers if m in text]
        return bool(hit), hit

    def model_metadata(self, *, model: ModelSpec) -> dict:
        """Return backend-specific metadata to embed in the JSON report.

        Default: backend name + model path basename. Overridable to capture
        commit hashes, library versions, GGUF metadata, etc.
        """
        return {
            "backend": self.name,
            "model": model.as_posix() if isinstance(model, Path) else model,
        }
