"""Render a pinned chat template locally so the server's template cannot vary.

The template comes from the hash-pinned ``tokenizer_config.json``. Metria
renders one user message with ``add_generation_prompt=True`` in a sandboxed
Jinja environment, tokenizes the rendered text without adding special tokens
again, and sends token IDs. Because Metria itself rendered it from pinned
bytes, the template is recorded with authority ``verified``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _token(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("content")
    return value if isinstance(value, str) else ""


def _template_source(config: Mapping[str, Any]) -> str:
    template = config.get("chat_template")
    if isinstance(template, list):
        named = {
            row.get("name"): row.get("template")
            for row in template
            if isinstance(row, Mapping)
        }
        template = named.get("default")
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError("the pinned tokenizer_config.json has no chat template")
    return template


class ChatTemplate:
    """A sandboxed, pinned chat template for single-turn user prompts."""

    def __init__(self, tokenizer_path: str, files: Mapping[str, str]) -> None:
        if "tokenizer_config.json" not in files:
            raise ValueError(
                "scenario.chat_template requires a pinned tokenizer_config.json"
            )
        path = Path(tokenizer_path).expanduser().resolve() / "tokenizer_config.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != files["tokenizer_config.json"]:
            raise RuntimeError("pinned tokenizer_config.json does not match")
        config = json.loads(raw)
        source = _template_source(config)
        try:
            from jinja2.sandbox import (  # type: ignore[import-not-found]
                ImmutableSandboxedEnvironment,
            )
        except ImportError as exc:
            raise RuntimeError(
                "scenario.chat_template needs the 'jinja2' package to render locally"
            ) from exc

        def raise_exception(message: str) -> None:
            raise RuntimeError(f"chat template error: {message}")

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        )
        environment.globals["raise_exception"] = raise_exception
        self._template = environment.from_string(source)
        self._tokens = {
            "bos_token": _token(config.get("bos_token")),
            "eos_token": _token(config.get("eos_token")),
        }
        self.sha256 = hashlib.sha256(source.encode()).hexdigest()

    def render(self, prompt: str) -> str:
        return str(
            self._template.render(
                messages=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                **self._tokens,
            )
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "status": "verified",
            "source": "pinned tokenizer_config.json rendered by Metria",
            "sha256": self.sha256,
            "messages": "single user turn with add_generation_prompt",
        }
