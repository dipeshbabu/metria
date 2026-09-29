"""Smoke-test an installed root distribution with an isolated Python interpreter.

Run with the target environment's Python and -I, from outside the source tree.
Only the standard library and installed Metria distribution are required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from importlib import resources
from importlib.metadata import distribution
from pathlib import Path

import metria


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    dist = distribution("metria")
    assert dist.metadata["Name"] == "metria"
    assert dist.version == metria.__version__ == args.version
    assert dist.metadata["License-Expression"] == "Apache-2.0"
    assert dist.metadata["Requires-Python"] == "<3.15,>=3.10"
    required = [value for value in (dist.requires or ()) if "extra ==" not in value]
    assert not required, "base package must not install an inference stack"
    from metria.fidelity import __report_schema__
    from metria.fidelity.measurement_math import approximate_topk_kl

    assert __report_schema__ == "kv_fidelity.report.v0.3.3"
    assert approximate_topk_kl({1: -0.5}, {1: -0.5}) == 0.0
    prompts = resources.files("metria.fidelity").joinpath("prompts/v0.1.jsonl")
    assert len(prompts.read_text(encoding="utf-8").splitlines()) == 30

    files = {str(path).replace("\\", "/") for path in (dist.files or ())}
    assert any(p.endswith(".dist-info/licenses/LICENSE") for p in files)
    assert any(p.endswith(".dist-info/licenses/NOTICE") for p in files)
    assert Path(metria.__file__).parent == Path(str(dist.locate_file("metria")))
    assert callable(metria.verify_recipe)
    assert callable(metria.execute_study)
    cli = Path(sys.executable).with_name(
        "metria.exe" if sys.platform == "win32" else "metria"
    )
    with tempfile.TemporaryDirectory(prefix="metria-install-") as directory:
        for arguments, expected in [
            (["--version"], f"metria {args.version}"),
            (["--help"], "verify"),
            (["verify", "--help"], "--output"),
            (["fidelity", "--help"], "score"),
            (["fidelity", "--version"], f"metria fidelity {args.version}"),
            (["fidelity", "compare", "--help"], "compare"),
            (["recipe", "--help"], "validate"),
            (["recipe", "prepare-vllm", "--help"], "--example"),
            (["recipe", "prepare-llamacpp-build", "--help"], "--reference-bin-dir"),
            (["recipe", "prepare-gguf-quantization", "--help"], "--candidate-model"),
            (["demo", "--help"], "not-comparable"),
            (["compare", "--help"], "compare"),
        ]:
            result = subprocess.run(
                [str(cli), *arguments],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
            assert expected in result.stdout, result.stdout
        _check_demos(cli, Path(directory))
    print(f"Installed Metria {args.version}: metadata, SDK, and CLI passed")
    return 0


def _check_demos(cli: Path, directory: Path) -> None:
    assets = resources.files("metria").joinpath("data")
    descriptor = json.loads(
        assets.joinpath("smollm2-135m.json").read_text(encoding="utf-8")
    )
    assert len(descriptor["files"]) == 8
    assert (
        len(
            assets.joinpath("prefix-workload.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 2
    )
    for case, code in (("pass", 0), ("fail", 1), ("not-comparable", 3)):
        output = directory / case
        result = subprocess.run(
            [str(cli), "demo", "--case", case, "--output", str(output)],
            cwd=directory,
            capture_output=True,
            text=True,
        )
        assert result.returncode == code, result.stderr
        data = json.loads((output / "verification.json").read_text(encoding="utf-8"))
        assert data["fixture_only"] is True and data["scope"] == "synthetic_fixture.v1"
        assert data["exit_code"] == code


if __name__ == "__main__":
    raise SystemExit(main())
