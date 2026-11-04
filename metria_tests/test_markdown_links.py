"""Repository link checks ignore local caches but cover new author documents."""

from __future__ import annotations

import subprocess

from tools.maintenance.check_markdown_links import check


def test_ignored_cache_links_do_not_break_repository_checks(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("cache/\n")
    (tmp_path / "README.md").write_text("[Guide](guide.md)\n")
    (tmp_path / "guide.md").write_text("A tracked guide.\n")
    subprocess.run(
        ["git", "add", ".gitignore", "README.md", "guide.md"], cwd=tmp_path, check=True
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "README.md").write_text("[Upstream fixture](missing.md)\n")
    assert check(tmp_path) == []
    (tmp_path / "new-guide.md").write_text("[Missing](absent.md)\n")
    assert check(tmp_path) == [
        (tmp_path.joinpath("new-guide.md").relative_to(tmp_path), "absent.md")
    ]


def test_deleted_tracked_link_target_is_still_reported(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("[Guide](guide.md)\n")
    (tmp_path / "guide.md").write_text("A guide.\n")
    subprocess.run(["git", "add", "README.md", "guide.md"], cwd=tmp_path, check=True)
    (tmp_path / "guide.md").unlink()
    assert check(tmp_path) == [
        (tmp_path.joinpath("README.md").relative_to(tmp_path), "guide.md")
    ]


def test_link_checks_work_outside_git_and_skip_deleted_sources(tmp_path):
    (tmp_path / "README.md").write_text("[Missing](absent.md)\n")
    assert check(tmp_path) == [
        (tmp_path.joinpath("README.md").relative_to(tmp_path), "absent.md")
    ]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").unlink()
    assert check(tmp_path) == []
