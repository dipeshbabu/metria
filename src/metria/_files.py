"""Publish complete sibling files without replacing existing directory entries."""

import os
from pathlib import Path


def _publish_new_file(temporary: Path, path: Path) -> None:
    if os.name == "nt":
        # Windows rename fails if any destination entry already exists.
        os.rename(temporary, path)
    else:
        os.link(temporary, path)
