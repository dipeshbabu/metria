"""Per-provider capture qualification for controlled llama.cpp build pairs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._freeze import freeze_mapping
from ..capture_support import probe_capture_support
from ..models import RunSpec
from ..protocols import CaptureRequest, RuntimeSession, SupportReport
from .llamacpp import (
    LlamaCppAdapter,
    LlamaCppSession,
    _check_pinned_model,
    _file_identity,
)

PROVIDERS_KEY = "llama_cpp_capture_providers"


class QualifiedLlamaCppAdapter(LlamaCppAdapter):
    """Bind each native capture provider to its own prequalified content pin."""

    def __init__(self, providers: Mapping[str, str]) -> None:
        self._providers = freeze_mapping(providers)

    def _check_provider(self, resolved: Mapping[str, Any]) -> None:
        runtime = resolved["runtime"]
        expected = self._providers[str(Path(runtime["bin_dir"]).resolve())]
        provider = runtime["completion"]
        if provider is None or provider.get("sha256") != expected:
            raise ValueError("resolved capture provider differs from its qualified pin")
        actual = _file_identity(Path(provider["path"]), include_hash=True)
        if actual["sha256"] != expected:
            raise ValueError("qualified capture provider changed after resolution")

    def probe_captures(
        self,
        spec: RunSpec,
        environment: Mapping[str, Any],
        capture: tuple[CaptureRequest, ...],
    ) -> SupportReport:
        adapter = LlamaCppAdapter()
        pin = self._providers[str(Path(spec.runtime["bin_dir"]).resolve())]
        return probe_capture_support(
            adapter=adapter,
            runtime_support=adapter.probe(spec, environment),
            spec=spec,
            environment={**environment, "llama_cpp_token_ids_capture_sha256": pin},
            capture=capture,
        )

    def resolve(
        self, spec: RunSpec, environment: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        resolved = super().resolve(spec, environment)
        self._check_provider(resolved)
        return resolved

    def launch(
        self, resolved: Mapping[str, Any], environment: Mapping[str, Any]
    ) -> RuntimeSession:
        self._check_provider(resolved)
        return super().launch(resolved, environment)

    def observe(self, session: RuntimeSession) -> Mapping[str, Any]:
        observed = super().observe(session)
        assert isinstance(session, LlamaCppSession)
        self._check_provider(session.resolved)
        _check_pinned_model(session.resolved["model"], hash_content=True)
        return observed
