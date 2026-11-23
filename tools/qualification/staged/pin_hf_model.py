#!/usr/bin/env python3
"""STAGED: download a Hugging Face model snapshot and write a pinned descriptor.

The descriptor records the resolved commit revision and the SHA-256 of every
file Metria needs, in the same shape as ``tools/qualification/vllm-*.json``.
Run it inside an environment with ``huggingface_hub`` (the vLLM venv has it)::

    python pin_hf_model.py Qwen/Qwen2.5-0.5B-Instruct --cache-dir hf \\
        --license apache-2.0 --output qwen.json --model-root-file qwen_root
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from huggingface_hub import (  # type: ignore[import-not-found]
        HfApi,
        snapshot_download,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root-file", type=Path, required=True)
    args = parser.parse_args()
    revision = HfApi().model_info(args.model_id, revision=args.revision).sha
    root = Path(
        snapshot_download(
            args.model_id,
            revision=revision,
            allow_patterns=list(FILES),
            cache_dir=str(args.cache_dir),
        )
    )
    files = {name: _sha256(root / name) for name in FILES if (root / name).is_file()}
    descriptor = {
        "model_id": args.model_id,
        "revision": revision,
        "license": f"{args.license} (upstream model metadata)",
        "files": files,
    }
    args.output.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    args.model_root_file.write_text(str(root), encoding="utf-8")
    print(json.dumps(descriptor, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
