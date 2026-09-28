"""Repeat a qualified reference/candidate recipe with retained verifier evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

source_dir = Path(__file__).resolve().parents[2] / "src"
if source_dir.is_dir():
    sys.path.insert(0, str(source_dir))

from metria import load_study_recipe  # noqa: E402
from metria.verification_trials import (  # noqa: E402
    VerificationTrialPolicy,
    execute_verification_trials,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup-pairs", type=int, default=0)
    parser.add_argument("--measured-pairs", type=int, default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline", type=Path)
    group.add_argument(
        "--no-ref",
        action="store_true",
        help="compatibility option: no cached baseline; recipe policy still applies",
    )
    args = parser.parse_args()
    try:
        baseline = (
            json.loads(args.baseline.read_text(encoding="utf-8"))
            if args.baseline
            else None
        )
        summary = execute_verification_trials(
            load_study_recipe(args.recipe),
            args.output,
            policy=VerificationTrialPolicy(args.warmup_pairs, args.measured_pairs),
            baseline=baseline,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "invalid_configuration",
                    "exit_code": 2,
                    "error_type": type(exc).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "exit_code": summary["exit_code"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
