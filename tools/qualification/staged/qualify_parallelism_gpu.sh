#!/bin/bash
# STAGED: native qualification of local_vllm_parallelism.v1 on a 2-GPU host.
#
# Reference on one GPU vs candidate with tensor parallel 2 (case tp2) and with
# pipeline parallel 2 (case pp2), through `metria recipe prepare-vllm` and
# `metria verify`. Uses Qwen2.5-0.5B-Instruct (14 attention heads, 2 KV heads,
# divisible by 2) pinned by pin_hf_model.py.
#
# Environment: WORK (directory with venv/ and metria-src/), OUT (new evidence
# directory), MODEL_ROOT (pinned snapshot), DESCRIPTOR (pinned descriptor).
set -u
WORK=${WORK:?}; OUT=${OUT:?}; MODEL_ROOT=${MODEL_ROOT:?}; DESCRIPTOR=${DESCRIPTOR:?}
export PYTHONPATH="$WORK/metria-src/src" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
mkdir -p "$OUT"
py="$WORK/venv/bin/python"

"$py" - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
prompts = [
    ("story", "Once upon a time"),
    ("dog", "The little dog"),
    ("capital", "The capital of France is"),
    ("arithmetic", "Two plus two equals"),
]
with open(f"{out}/workload.jsonl", "w") as f:
    for pid, prompt in prompts:
        f.write(json.dumps({"id": pid, "prompt": prompt, "checks": [
            {"id": "answered", "kind": "nonempty"},
        ]}) + "\n")
json.dump({"schema": "metria.verification_policy.v1", "criteria": [
    {"target": "quality.pass_rate_delta", "version": "1", "min": 0.0},
    {"target": "behavior.trajectory_agreement", "version": "0.3.4", "min": 0.9},
    {"target": "behavior.trajectory_kld_mean", "version": "0.1.0", "max": 0.05},
]}, open(f"{out}/policy.json", "w"), indent=2)
PY

run_case() {  # name option
  name=$1; shift
  "$py" -m metria recipe prepare-vllm --model "$MODEL_ROOT" --descriptor "$DESCRIPTOR" \
    --workload "$OUT/workload.jsonl" --policy "$OUT/policy.json" "$@" --staged-kld \
    --max-tokens 16 --measured-trials 2 --output "$OUT/$name.study.json" > "$OUT/$name.log" 2>&1
  "$py" -m metria verify "$OUT/$name.study.json" --output "$OUT/$name" >> "$OUT/$name.log" 2>&1
  echo "$name exit=$?"
  grep -E "Verdict|>=|<=|Observed tensor" "$OUT/$name.log" | head -12
}

run_case tp2 --candidate-tensor-parallel-size 2
run_case pp2 --candidate-pipeline-parallel-size 2
echo PARALLEL_DONE
