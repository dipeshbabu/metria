#!/usr/bin/env bash
# Compatibility entry point for bounded, evidence-backed verification trials.
# See docs/guides/verification-trials.md for migration from legacy arguments.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$script_dir/verification_trials.py" "$@"
