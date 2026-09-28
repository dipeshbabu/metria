from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from packaging.requirements import Requirement


def _check(data):
    module = runpy.run_path(
        str(
            Path(__file__).parents[1] / "tools/maintenance/check_public_dependencies.py"
        )
    )
    return module["compatible_releases"](Requirement("metria>=0.1.1.dev0,<0.2"), data)


def test_unpublished_prerelease_and_yanked_dependencies_block_publication():
    assert (
        _check(
            {
                "releases": {
                    "0.1.0": [{"yanked": False}],
                    "0.1.1.dev0": [{"yanked": False}],
                    "0.1.1": [{"yanked": True}],
                    "0.2.0": [{"yanked": False}],
                }
            }
        )
        == ()
    )


def test_stable_compatible_release_requires_at_least_one_non_yanked_file():
    assert _check(
        {
            "releases": {
                "0.1.2": [{"yanked": True}, {"yanked": False}],
                "0.1.1": [],
                "0.1.3": [{}],
            }
        }
    ) == ("0.1.2",)


def test_malformed_index_response_is_not_success():
    with pytest.raises(ValueError, match="does not contain releases"):
        _check({})
