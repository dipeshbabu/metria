"""Check every rendered local page, asset, fragment, and the site search index."""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "id" or (tag == "a" and name == "name"):
                self.ids.add(value)
            if name in {"href", "src", "poster"}:
                self.links.append(value)


def check_site(directory: Path, base_path: str = "/metria/") -> list[str]:
    root = directory.resolve()
    pages = {
        path: Document(path.read_text(encoding="utf-8"))
        for path in root.rglob("*.html")
    }
    errors = []
    if root / "index.html" not in pages:
        errors.append("site has no index.html")
    sources = {source: page.links for source, page in pages.items()}
    for source in root.rglob("*.css"):
        sources[source] = re.findall(
            r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", source.read_text(encoding="utf-8")
        )
    for source, links in sources.items():
        for link in links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            value = unquote(url.path)
            if value.startswith("/"):
                if not value.startswith(base_path):
                    errors.append(
                        f"{source.relative_to(root)}: link escapes base path: {link}"
                    )
                    continue
                target = root / value[len(base_path) :]
            else:
                target = source.parent / value if value else source
            target = target.resolve()
            if not target.is_relative_to(root):
                errors.append(f"{source.relative_to(root)}: link escapes site: {link}")
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{source.relative_to(root)}: missing target: {link}")
            elif url.fragment and target in pages:
                if unquote(url.fragment) not in pages[target].ids:
                    errors.append(
                        f"{source.relative_to(root)}: missing fragment: {link}"
                    )
    search = root / "search" / "search_index.json"
    try:
        data = json.loads(search.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("docs"), list)
            or not data["docs"]
        ):
            errors.append("search index has no documents")
    except (OSError, ValueError):
        errors.append("search index is missing or invalid")
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    errors = check_site(args.directory)
    if errors:
        print("\n".join(errors))
        return 1
    count = len(list(args.directory.rglob("*.html")))
    print(
        f"Validated {count} HTML pages, local links/assets/fragments, and search index."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
