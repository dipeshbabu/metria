# /// script
# requires-python = ">=3.11"
# dependencies = ["mkdocs-material==9.8.1"]
# ///
"""Run the documentation builder in its own locked environment."""

from importlib import import_module

if __name__ == "__main__":
    import_module("mkdocs.__main__").cli()
