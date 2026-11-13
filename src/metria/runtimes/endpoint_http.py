"""Credential-safe HTTP transport for OpenAI-compatible inference endpoints.

URLs and API keys are read only from process environment variables named by
the recipe. Neither value is returned in evidence: callers receive the parsed
endpoint's safe identity fields and per-attempt status codes and timings.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Status 0 is a transport failure (refused, reset, timeout).
_RETRY_STATUS = frozenset({0, 429, 500, 502, 503, 504})
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_backoff = time.sleep
_ENV_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class EndpointError(RuntimeError):
    """A request failed; the message never contains the URL, key or body."""


def _env_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or set(value) - _ENV_NAME_CHARS
        or value[0].isdigit()
    ):
        raise ValueError(f"{label} must name an upper-case environment variable")
    return value


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_base_url(value: str) -> urllib.parse.SplitResult:
    """Accept https anywhere and http only on loopback; reject embedded secrets."""

    parts = urllib.parse.urlsplit(value.strip())
    host = parts.hostname or ""
    if parts.scheme not in {"https", "http"} or not host:
        raise ValueError("endpoint URL must be an absolute http(s) URL")
    if parts.scheme == "http" and not _loopback(host):
        raise ValueError("plain http is allowed only for loopback endpoints")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "endpoint URL must not contain credentials, a query or a fragment"
        )
    return parts


@dataclass
class EndpointConnection:
    """One endpoint resolved from the environment for the lifetime of a run."""

    base: urllib.parse.SplitResult
    api_key: str | None
    timeout_s: float
    max_retries: int
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_environment(
        cls,
        base_url_env: str,
        api_key_env: str | None,
        *,
        timeout_s: float,
        max_retries: int,
    ) -> EndpointConnection:
        url = os.environ.get(_env_name(base_url_env, "base_url_env"))
        if not url:
            raise ValueError(f"environment variable {base_url_env} is not set")
        key = None
        if api_key_env is not None:
            key = os.environ.get(_env_name(api_key_env, "api_key_env"))
            if not key:
                raise ValueError(f"environment variable {api_key_env} is not set")
        return cls(parse_base_url(url), key, timeout_s, max_retries)

    def identity(self) -> dict[str, Any]:
        """Safe endpoint fields accepted by ``RuntimeIdentityEvidence``."""

        host = self.base.hostname or ""
        result: dict[str, Any] = {
            "scheme": self.base.scheme,
            "port": self.base.port or (443 if self.base.scheme == "https" else 80),
            "path_sha256": hashlib.sha256(self.base.path.encode()).hexdigest(),
        }
        # Private hostnames stay out of durable evidence; loopback is not private.
        if _loopback(host):
            result["host"] = host
        return result

    def _url(self, path: str) -> str:
        return urllib.parse.urlunsplit(
            (
                self.base.scheme,
                self.base.netloc,
                self.base.path.rstrip("/") + path,
                "",
                "",
            )
        )

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> tuple[Any, float]:
        """Send one JSON request with bounded retries; return (payload, seconds)."""

        data = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter_ns()
            status, payload = self._send(method, path, data, headers)
            seconds = (time.perf_counter_ns() - started) / 1_000_000_000
            self.attempts.append(
                {"path": path, "attempt": attempt, "status": status, "seconds": seconds}
            )
            if status == 200:
                return payload, seconds
            if status not in _RETRY_STATUS or attempt == self.max_retries:
                break
            _backoff(min(0.5 * 2**attempt, 8.0))
        raise EndpointError(f"{method} {path} failed with status {status}")

    def _send(
        self, method: str, path: str, data: bytes | None, headers: dict[str, str]
    ) -> tuple[int, Any]:
        request = urllib.request.Request(
            self._url(path), data=data, headers=headers, method=method
        )
        context = ssl.create_default_context()
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme validated above
                request, timeout=self.timeout_s, context=context
            ) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                status = response.status
        except urllib.error.HTTPError as exc:
            return exc.code, None
        except (urllib.error.URLError, OSError, TimeoutError):
            # Status 0 marks a transport failure; its text may name the host.
            return 0, None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise EndpointError(f"{method} {path} response exceeds the size bound")
        try:
            return status, json.loads(raw)
        except ValueError:
            raise EndpointError(f"{method} {path} returned invalid JSON") from None


def bounded_timeout(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 3600
    ):
        raise ValueError("timeout_s must be a positive number of at most 3600")
    return float(value)


def bounded_retries(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise ValueError("max_retries must be an integer from 0 to 5")
    return value
