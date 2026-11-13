"""Keep repository Markdown links useful when the same docs become a website."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REPOSITORY = "https://github.com/dipeshbabu/metria"
_LINK = re.compile(r"(?P<prefix>!?\[[^\n]*?\]\()(?P<url><[^>]+>|[^\s)]+)")
_REFERENCE = re.compile(r"^(?P<prefix> {0,3}\[[^\]]+\]:\s*)(?P<url><[^>]+>|\S+)")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def on_config(config: Any) -> Any:
    config.extra["source_ref"] = os.environ.get("GITHUB_SHA", "main")
    return config


def repository_link(url: str, source: Path, ref: str = "main") -> str:
    """Leave site links local; send links outside docs to the matching Git tree."""
    target = urlsplit(url)
    if target.scheme or target.netloc or not target.path or target.path.startswith("/"):
        return url
    resolved = (source.parent / unquote(target.path)).resolve()
    if resolved.is_relative_to(DOCS):
        return url
    if not resolved.is_relative_to(ROOT) or not resolved.exists():
        raise ValueError(f"Documentation link has no repository target: {url}")
    path = quote(resolved.relative_to(ROOT).as_posix(), safe="/")
    if resolved.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        base = f"https://raw.githubusercontent.com/dipeshbabu/metria/{ref}/{path}"
    else:
        kind = "tree" if resolved.is_dir() else "blob"
        base = f"{REPOSITORY}/{kind}/{ref}/{path}"
    result = urlsplit(base)
    return urlunsplit(
        (result.scheme, result.netloc, result.path, target.query, target.fragment)
    )


def on_page_markdown(markdown: str, *, page: Any, config: Any, **kwargs: Any) -> str:
    source = Path(page.file.abs_src_path)
    ref = config.extra.get("source_ref", "main")

    def replace_link(match: re.Match[str]) -> str:
        url = match["url"]
        angled = url.startswith("<")
        updated = repository_link(url[1:-1] if angled else url, source, ref)
        return match["prefix"] + (f"<{updated}>" if angled else updated)

    lines = []
    fence = ""
    for line in markdown.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            value = marker[1]
            if not fence:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = ""
            lines.append(line)
        elif fence:
            lines.append(line)
        else:
            lines.append(_REFERENCE.sub(replace_link, _LINK.sub(replace_link, line)))
    return "".join(lines)
