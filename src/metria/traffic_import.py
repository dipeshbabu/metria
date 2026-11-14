"""Import a redacted, deterministic sample of production prompts as a workload.

``metria recipe import-traffic`` reads a JSONL request log, extracts one prompt
per row, samples rows with a recorded seed, redacts secrets and personal data,
and writes a workload JSONL plus a provenance manifest. Original prompt text is
never written. Only redacted prompts and their hashes are retained, and the
manifest records how many replacements each rule made, never what was replaced.

Redaction is best effort. After redaction every rule runs again, and the import
refuses to write anything if any rule still matches. Literal secrets supplied
with ``--secret-file`` are both replaced and verified absent.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

SCHEMA = "metria.traffic_import.v1"
MAX_PROMPT_CHARS = 32_768
MAX_SOURCE_BYTES = 512 * 1024 * 1024

# Order matters: specific credential formats before generic shapes.
BUILTIN_RULES: tuple[tuple[str, str], ...] = (
    (
        "private_key",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    ),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ("bearer_token", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    ("openai_key", r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    ("aws_access_key", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    (
        "github_token",
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b",
    ),
    ("slack_token", r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    ("url_credentials", r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@[^\s]+"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("ipv4", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ("card_number", r"\b(?:\d[ -]?){13,19}\b"),
    (
        "phone",
        r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?![\w.])",
    ),
    (
        "long_secret",
        r"\b(?=[A-Za-z0-9+/_-]*\d)(?=[A-Za-z0-9+/_-]*[A-Za-z])[A-Za-z0-9+/_-]{40,}={0,2}",
    ),
)


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


class Redactor:
    """Apply named rules and literal secrets; count replacements per rule."""

    def __init__(
        self,
        extra: Sequence[tuple[str, str]] = (),
        secrets: Iterable[str] = (),
    ) -> None:
        self.rules = [
            (name, re.compile(pattern)) for name, pattern in (*BUILTIN_RULES, *extra)
        ]
        self.secrets = sorted({item for item in secrets if item}, key=len, reverse=True)
        self.counts: dict[str, int] = {}

    def _replace(self, name: str, pattern: re.Pattern[str], text: str) -> str:
        def substitute(match: re.Match[str]) -> str:
            if name == "card_number":
                digits = re.sub(r"\D", "", match.group(0))
                if not 13 <= len(digits) <= 19 or not _luhn(digits):
                    return match.group(0)
            self.counts[name] = self.counts.get(name, 0) + 1
            return f"[REDACTED:{name.upper()}]"

        return pattern.sub(substitute, text)

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                self.counts["secret_file"] = self.counts.get(
                    "secret_file", 0
                ) + text.count(secret)
                text = text.replace(secret, "[REDACTED:SECRET]")
        for name, pattern in self.rules:
            text = self._replace(name, pattern, text)
        return text

    def residual(self, text: str) -> list[str]:
        """Rules that still match after redaction (placeholders excluded)."""

        cleaned = re.sub(r"\[REDACTED:[A-Z_]+\]", " ", text)
        found = [secret for secret in self.secrets if secret in cleaned]
        hits = ["secret_file"] if found else []
        for name, pattern in self.rules:
            for match in pattern.finditer(cleaned):
                if name == "card_number":
                    digits = re.sub(r"\D", "", match.group(0))
                    if not 13 <= len(digits) <= 19 or not _luhn(digits):
                        continue
                hits.append(name)
                break
        return hits


def _field(row: Any, path: str) -> Any:
    value = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def extract_prompt(row: Any, field: str | None) -> str | None:
    """Take an explicit field, else ``prompt``, ``input`` or the last user message."""

    if field:
        value = _field(row, field)
        return value if isinstance(value, str) else None
    if not isinstance(row, Mapping):
        return None
    for key in ("prompt", "input"):
        if isinstance(row.get(key), str):
            return row[key]
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return message["content"]
    return None


def import_traffic(
    source: Path,
    *,
    sample: int,
    seed: int,
    field: str | None = None,
    extra_rules: Sequence[tuple[str, str]] = (),
    secrets: Iterable[str] = (),
    checks: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return redacted workload rows and a provenance manifest."""

    if not 1 <= sample <= 10_000:
        raise ValueError("sample must be from 1 to 10000 rows")
    raw = source.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("traffic log exceeds the 512 MiB bound")
    candidates: list[str] = []
    skipped = {"invalid_json": 0, "no_prompt": 0, "too_long": 0}
    lines = raw.decode("utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            skipped["invalid_json"] += 1
            continue
        prompt = extract_prompt(record, field)
        if not prompt or not prompt.strip():
            skipped["no_prompt"] += 1
        elif len(prompt) > MAX_PROMPT_CHARS:
            skipped["too_long"] += 1
        else:
            candidates.append(prompt)
    if not candidates:
        raise ValueError("no usable prompts were found in the traffic log")
    chosen = sorted(
        random.Random(seed).sample(range(len(candidates)), min(sample, len(candidates)))
    )
    redactor = Redactor(extra_rules, secrets)
    rows = []
    for number, index in enumerate(chosen, start=1):
        text = redactor.redact(candidates[index])
        if hits := redactor.residual(text):
            raise ValueError(
                "redaction left matches for: "
                + ", ".join(sorted(set(hits)))
                + "; nothing was written"
            )
        row: dict[str, Any] = {
            "id": f"traffic-{number:05d}",
            "category": "traffic",
            "prompt": text,
        }
        if checks:
            row["checks"] = [dict(check) for check in checks]
        rows.append(row)
    manifest = {
        "schema": SCHEMA,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": sum(1 for line in lines if line.strip()),
        "usable_rows": len(candidates),
        "skipped": skipped,
        "selected": len(rows),
        "seed": seed,
        "field": field or "prompt | input | last user message",
        "rules": [name for name, _ in (*BUILTIN_RULES, *extra_rules)],
        "secret_file_entries": len(redactor.secrets),
        "redactions": dict(sorted(redactor.counts.items())),
        "prompt_sha256": [
            hashlib.sha256(row["prompt"].encode()).hexdigest() for row in rows
        ],
        "limitation": "Pattern-based redaction is best effort; review the workload before sharing it.",
    }
    return rows, manifest


def _rule(text: str) -> tuple[str, str]:
    name, separator, pattern = text.partition("=")
    if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
        raise ValueError("--redact-pattern uses NAME=REGEX with a lowercase name")
    re.compile(pattern)
    return name, pattern


def add_traffic_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "import-traffic",
        help="sample and redact production prompts into a workload JSONL",
    )
    parser.add_argument("--source", type=Path, required=True, help="request log JSONL")
    parser.add_argument("--output", type=Path, required=True, help="new workload JSONL")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="new provenance JSON"
    )
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--field", help="dotted path to the prompt in each row")
    parser.add_argument(
        "--redact-pattern", action="append", default=[], metavar="NAME=REGEX"
    )
    parser.add_argument(
        "--secret-file",
        type=Path,
        help="file with one literal secret per line to replace and verify absent",
    )
    parser.add_argument(
        "--nonempty-check",
        action="store_true",
        help="attach a nonempty task check to every prompt",
    )


def import_traffic_command(args: Any, stdout: TextIO) -> int:
    from .verification import _write_atomic

    for path in (args.output, args.manifest):
        if path.exists():
            raise FileExistsError(f"{path.name} already exists; choose a new file")
    secrets = []
    if args.secret_file is not None:
        secrets = [
            line.strip()
            for line in args.secret_file.read_text(encoding="utf-8").splitlines()
        ]
    rows, manifest = import_traffic(
        args.source,
        sample=args.sample,
        seed=args.seed,
        field=args.field,
        extra_rules=[_rule(item) for item in args.redact_pattern],
        secrets=secrets,
        checks=[{"id": "answered", "kind": "nonempty"}] if args.nonempty_check else (),
    )
    _write_atomic(args.output, "".join(json.dumps(row) + "\n" for row in rows))
    _write_atomic(args.manifest, json.dumps(manifest, indent=2) + "\n")
    stdout.write(
        f"Imported {manifest['selected']} of {manifest['usable_rows']} usable prompts "
        f"into {args.output}\nRedactions: {manifest['redactions'] or 'none'}\n"
        "Review the workload before sharing it; redaction is best effort.\n"
    )
    return 0
