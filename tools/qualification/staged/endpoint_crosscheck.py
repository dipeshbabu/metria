#!/usr/bin/env python3
"""STAGED: compare endpoint token IDs with an in-process vLLM run record.

The endpoint profile's reference run (trial 0) and an in-process run of the
same pinned model, prompts and greedy decode settings should produce identical
token IDs if the HTTP path adds no tokenization or decoding difference.

Usage::

    python3 endpoint_crosscheck.py ENDPOINT_REFERENCE.run.json IN_PROCESS.run.json \\
        --output crosscheck.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRAJECTORY = "kv_fidelity.decode_time_trajectory"


def _tokens(path: Path, *, trial_prefix: str | None) -> dict[str, list[int]]:
    record = json.loads(path.read_text(encoding="utf-8"))["record"]
    rows = record["evidence"]["measurements"][TRAJECTORY]["prompts"]
    result = {}
    for row in rows:
        prompt_id = row["id"]
        if trial_prefix is not None:
            prefix, _, prompt_id = prompt_id.partition(":")
            if prefix != trial_prefix:
                continue
        result[prompt_id] = list(row["token_ids"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("endpoint_record", type=Path)
    parser.add_argument("in_process_record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoint = _tokens(args.endpoint_record, trial_prefix="0")
    local = _tokens(args.in_process_record, trial_prefix=None)
    shared = sorted(set(endpoint) & set(local))
    rows: list[dict[str, Any]] = [
        {
            "id": prompt_id,
            "matched": endpoint[prompt_id] == local[prompt_id],
            "endpoint_tokens": len(endpoint[prompt_id]),
            "in_process_tokens": len(local[prompt_id]),
        }
        for prompt_id in shared
    ]
    summary = {
        "schema": "metria.endpoint_crosscheck.v1",
        "compared_prompts": len(rows),
        "all_matched": bool(rows) and all(row["matched"] for row in rows),
        "per_prompt": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
