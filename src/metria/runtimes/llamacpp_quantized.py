"""Bind native llama.cpp runs to inspected immutable GGUF quantization artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._freeze import freeze_mapping
from ..models import RunSpec
from ..protocols import RuntimeSession
from .gguf_identity import inspect_gguf
from .llamacpp import LlamaCppSession
from .llamacpp_qualified import QualifiedLlamaCppAdapter


class QuantizedLlamaCppAdapter(QualifiedLlamaCppAdapter):
    def __init__(
        self, providers: Mapping[str, str], expected: Mapping[str, Any]
    ) -> None:
        super().__init__(providers)
        self._expected = freeze_mapping(expected)

    def _inspect(self, resolved: Mapping[str, Any]) -> dict[str, Any]:
        model = resolved["model"]
        observed = inspect_gguf(model["path"])
        if observed != self._expected.get(model["sha256"]):
            raise ValueError(
                "GGUF metadata does not match the pinned conversion manifest"
            )
        return observed

    def resolve(
        self, spec: RunSpec, environment: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        resolved = super().resolve(spec, environment)
        return freeze_mapping({**resolved, "gguf": self._inspect(resolved)})

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]:
        observed = super().observe(session)
        assert isinstance(session, LlamaCppSession)
        gguf = self._inspect(session.resolved)
        return freeze_mapping({**observed, "gguf": gguf})
