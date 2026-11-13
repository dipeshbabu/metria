"""Keep published navigation and repository evidence links valid."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(name):
    path = Path(__file__).resolve().parents[1] / "tools" / "docs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"docs_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hooks = _module("hooks")
checker = _module("check_site")


def test_evidence_links_keep_revision_and_fragment_while_docs_stay_local():
    source = hooks.DOCS / "guides" / "getting-started.md"
    assert hooks.repository_link("../index.md#start", source) == "../index.md#start"
    assert hooks.repository_link("#start", source) == "#start"
    assert (
        hooks.repository_link("https://example.org/", source) == "https://example.org/"
    )
    assert hooks.repository_link(
        "../../examples/verification/README.md?plain=1#read", source, "abc123"
    ) == (
        "https://github.com/dipeshbabu/metria/blob/abc123/examples/verification/README.md?plain=1#read"
    )


def test_rewriting_preserves_code_fences_and_supports_reference_links():
    source = hooks.DOCS / "guides" / "getting-started.md"
    page = SimpleNamespace(file=SimpleNamespace(abs_src_path=str(source)))
    config = SimpleNamespace(extra={"source_ref": "abc123"})
    code = "```markdown\n[Example](../../README.md)\n```\n"
    text = code + "[Example](../../README.md)\n[repo]: ../../README.md#read\n"
    output = hooks.on_page_markdown(text, page=page, config=config)
    assert output.startswith(code)
    assert (
        output.count("https://github.com/dipeshbabu/metria/blob/abc123/README.md") == 2
    )


def test_missing_repository_target_fails_the_build():
    with pytest.raises(ValueError, match="no repository target"):
        hooks.repository_link(
            "../../missing-document.md", hooks.DOCS / "guides" / "a.md"
        )


@pytest.mark.parametrize(
    "text", ["one\u2014two", "one &mdash; two", "one &#8212; two", "one &#x2014; two"]
)
def test_documentation_rejects_em_dashes(text):
    source = hooks.DOCS / "guides" / "a.md"
    page = SimpleNamespace(file=SimpleNamespace(abs_src_path=str(source)))
    with pytest.raises(ValueError, match="instead of em dashes"):
        hooks.on_page_markdown(text, page=page, config=SimpleNamespace(extra={}))


def test_documentation_keeps_compound_word_hyphens():
    source = hooks.DOCS / "guides" / "a.md"
    page = SimpleNamespace(file=SimpleNamespace(abs_src_path=str(source)))
    text = "A reference-candidate comparison with end-to-end evidence.\n"
    assert (
        hooks.on_page_markdown(text, page=page, config=SimpleNamespace(extra={}))
        == text
    )


def _site(tmp_path):
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "search_index.json").write_text(
        json.dumps({"docs": [{"title": "Start"}]})
    )
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide" / "index.html").write_text(
        '<h1 id="install">Install</h1><a href="../">Home</a>'
    )
    (tmp_path / "style.css").write_text("body{}")
    (tmp_path / "index.html").write_text(
        '<a href="guide/#install">Guide</a><link href="/metria/style.css">'
    )
    return tmp_path


def test_rendered_site_accepts_nested_navigation_base_path_and_fragments(tmp_path):
    assert checker.check_site(_site(tmp_path)) == []


@pytest.mark.parametrize(
    "link", ["missing.css", "guide/#missing", "/wrong-base/", "../outside.html"]
)
def test_rendered_site_rejects_broken_assets_fragments_and_escaping_links(
    tmp_path, link
):
    root = _site(tmp_path)
    (root / "index.html").write_text(f'<a href="{link}">Broken</a>')
    assert checker.check_site(root)


def test_rendered_site_requires_search_data(tmp_path):
    root = _site(tmp_path)
    (root / "search" / "search_index.json").unlink()
    assert checker.check_site(root) == ["search index is missing or invalid"]


def test_stylesheet_icon_references_are_checked(tmp_path):
    root = _site(tmp_path)
    (root / "style.css").write_text(".icon { mask-image: url('icon.svg'); }")
    assert checker.check_site(root) == ["style.css: missing target: icon.svg"]
    (root / "icon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    assert checker.check_site(root) == []
