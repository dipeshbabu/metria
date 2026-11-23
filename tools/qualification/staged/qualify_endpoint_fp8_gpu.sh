#!/bin/bash
# STAGED: GPU qualification of remote_endpoint_change.v1 (automatic vs FP8 KV).
#
# Starts two `vllm serve` processes from the pinned CUDA wheel, one per GPU:
# the reference with the automatic KV cache and the candidate with an FP8 KV
# cache. Then runs `metria recipe prepare-endpoint` with declared
# kv_cache_dtype auto -> fp8 and staged top-20 distributions, and
# `metria verify` with a trajectory-KLD policy.
#
# Environment: WORK (directory with venv/, metria-src/ and model_root),
# OUT (new evidence directory).
set -u
WORK=${WORK:?}; OUT=${OUT:?}
cd "$WORK"; model=$(cat model_root)
export PYTHONPATH="$WORK/metria-src/src" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
mkdir -p "$OUT"
tok=/opt/metria-endpoint-qual/tokenizer; mkdir -p "$tok"
cp "$model"/tokenizer.json "$model"/tokenizer_config.json "$model"/special_tokens_map.json \
  "$model"/vocab.json "$model"/merges.txt "$tok"/ 2>/dev/null

serve() {  # gpu port kv log
  CUDA_VISIBLE_DEVICES=$1 venv/bin/vllm serve "$model" --served-model-name smollm2 \
    --dtype float16 --max-model-len 512 --enforce-eager --max-num-seqs 1 \
    --gpu-memory-utilization 0.5 --kv-cache-dtype "$3" --host 127.0.0.1 --port "$2" > "$4" 2>&1 &
  echo $!
}
wait_ready() {
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 2; done
  return 1
}
ref=$(serve 0 8901 auto "$OUT/serve-reference.log")
cand=$(serve 1 8902 fp8 "$OUT/serve-candidate.log")
wait_ready 8901 && wait_ready 8902 || { echo "servers did not start"; kill "$ref" "$cand"; exit 1; }

venv/bin/python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
rows = [("story", "Once upon a time"), ("dog", "The little dog"),
        ("capital", "The capital of France is"), ("arithmetic", "Two plus two equals")]
with open(f"{out}/workload.jsonl", "w") as f:
    for pid, prompt in rows:
        f.write(json.dumps({"id": pid, "prompt": prompt,
                            "checks": [{"id": "answered", "kind": "nonempty"}]}) + "\n")
json.dump({"schema": "metria.verification_policy.v1", "criteria": [
    {"target": "quality.candidate_pass_rate", "version": "1", "min": 1.0},
    {"target": "behavior.trajectory_kld_mean", "version": "0.1.0", "max": 0.5},
]}, open(f"{out}/policy.json", "w"), indent=2)
PY
export METRIA_REFERENCE_URL=http://127.0.0.1:8901 METRIA_CANDIDATE_URL=http://127.0.0.1:8902
venv/bin/python -m metria recipe prepare-endpoint --model-id smollm2 --tokenizer "$tok" \
  --workload "$OUT/workload.jsonl" --policy "$OUT/policy.json" \
  --reference-url-env METRIA_REFERENCE_URL --candidate-url-env METRIA_CANDIDATE_URL \
  --reference-declared kv_cache_dtype=auto --candidate-declared kv_cache_dtype=fp8 \
  --reference-declared engine_version=0.30.0 --candidate-declared engine_version=0.30.0 \
  --max-tokens 16 --measured-trials 2 --staged-kld --timeout 300 \
  --output "$OUT/fp8.study.json" > "$OUT/fp8.log" 2>&1
venv/bin/python -m metria verify "$OUT/fp8.study.json" --output "$OUT/fp8" >> "$OUT/fp8.log" 2>&1
echo "fp8 exit=$?"
grep -E "Verdict|>=|<=|Claimed" "$OUT/fp8.log" | head -10
grep -E "kv_cache_dtype|fp8" "$OUT/serve-candidate.log" | head -3 | cut -c1-240
kill "$ref" "$cand"; wait 2>/dev/null
echo ENDPOINT_FP8_DONE
