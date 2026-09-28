"""Prepare a pinned prefix-cache recipe inside the installed vLLM environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

source = Path(__file__).resolve().parents[2] / "src"
if source.is_dir():
    sys.path.insert(0, str(source))

from metria.preparation import prepare_vllm_prefix_recipe  # noqa: E402
from metria.recipes import study_recipe_to_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument(
        "--workload", required=True, type=Path, help="JSONL id/prompt rows"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    descriptor = json.loads(args.descriptor.read_text(encoding="utf-8"))
    prompts = [
        json.loads(line)
        for line in args.workload.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recipe = prepare_vllm_prefix_recipe(
        args.model,
        descriptor,
        prompts,
        context=args.context,
        max_tokens=args.max_tokens,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        timeout_s=args.timeout,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(study_recipe_to_json(recipe) + "\n")
    print(f"Prepared pinned prefix-cache verification: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
