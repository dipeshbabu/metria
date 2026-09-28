"""Append a canonical verifier report to a provider-specific summary file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from metria.verification import VERIFICATION_SCHEMA, render_verification


def append_summary(output: Path, destination: Path) -> None:
    canonical = output / "verification.json"
    if canonical.is_file():
        data = json.loads(canonical.read_text(encoding="utf-8"))
        if data.get("schema") != VERIFICATION_SCHEMA:
            raise ValueError("unsupported verification result schema")
        report = render_verification(data)
    else:
        report = (
            "# Metria Verification\n\n"
            "No complete verification bundle was published. Inspect the verifier "
            "step and any retained run records for input, execution, or persistence failures.\n"
        )
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(report + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--summary-file", type=Path, default=os.environ.get("GITHUB_STEP_SUMMARY")
    )
    args = parser.parse_args()
    if args.summary_file is None:
        parser.error("set GITHUB_STEP_SUMMARY or pass --summary-file")
    append_summary(args.output, args.summary_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
