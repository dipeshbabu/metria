"""Exercise the unified command/API and the source-only compatibility bridge."""

from importlib import import_module, resources
from io import StringIO

import pytest

from metria import __version__
from metria.cli import main
from metria.fidelity.measurement_math import approximate_topk_kl
from metria.fidelity.report import _looks_like_kv_fidelity_cli


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["--help"], "score"),
        (["--version"], f"metria fidelity {__version__}"),
        (["score", "--help"], "--candidate"),
        (["compare", "--help"], "compare"),
        (["selftest", "--help"], "selftest"),
        (["repeatability", "--help"], "--runs"),
    ],
)
def test_fidelity_commands_use_the_metria_frontend(arguments, expected):
    output = StringIO()
    assert main(["fidelity", *arguments], stdout=output, stderr=StringIO()) == 0
    assert expected in output.getvalue()


def test_legacy_source_imports_resolve_to_the_same_implementation():
    for suffix in (
        "measurement_math",
        "contracts",
        "backends.base",
        "axes.trajectory",
        "cli",
    ):
        legacy = import_module("kv_fidelity." + suffix)
        canonical = import_module("metria.fidelity." + suffix)
        assert legacy is canonical
    assert approximate_topk_kl({1: -0.5}, {1: -0.5}) == 0.0


def test_prompt_assets_belong_to_the_public_distribution():
    prompts = resources.files("metria.fidelity").joinpath("prompts/v0.1.jsonl")
    assert len(prompts.read_text(encoding="utf-8").splitlines()) == 30


@pytest.mark.parametrize(
    "argv",
    [
        ["metria", "fidelity", "score"],
        ["C:\\tools\\metria.exe", "fidelity", "score"],
        ["/app/metria/__main__.py", "fidelity", "score"],
    ],
)
def test_unified_command_is_identified_for_sanitized_reproduction(argv):
    assert _looks_like_kv_fidelity_cli(argv)
