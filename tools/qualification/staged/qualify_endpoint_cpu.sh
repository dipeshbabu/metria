#!/bin/bash
# STAGED: native CPU qualification of remote_endpoint_change.v1 with vllm serve.
#
# Starts a float32 reference server and, in turn, candidate servers for each
# case, then runs `metria recipe prepare-endpoint` and `metria verify`:
#   identical    float32 vs float32, nothing declared differs     -> expect PASS
#   bf16         float32 vs bfloat16, declared dtype differs       -> policy decides
#   contradiction declared candidate engine_version 0.29.0         -> EXECUTION_FAILED
#   undeclared   candidate --max-model-len 512, not declared       -> NOT_COMPARABLE
#
# Environment: WORK (vLLM CPU venv directory with model_root), METRIA_SRC
# (checkout with src/ and tools/), OUT (new evidence directory).
set -u
WORK=${WORK:?}; METRIA_SRC=${METRIA_SRC:?}; OUT=${OUT:?}
cd "$WORK"
model=$(cat model_root)
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 OMP_NUM_THREADS=2 \
  VLLM_CPU_KVCACHE_SPACE=1
export PYTHONPATH="$METRIA_SRC/src" PYTHONDONTWRITEBYTECODE=1
rm -rf "$OUT"; mkdir -p "$OUT"
tok="${TOKENIZER_DIR:-/opt/metria-endpoint-qual/tokenizer}"; mkdir -p "$tok"
cp "$model"/tokenizer.json "$model"/tokenizer_config.json "$model"/special_tokens_map.json "$tok"/ 2>/dev/null
cp "$model"/vocab.json "$model"/merges.txt "$tok"/ 2>/dev/null

serve() {  # port cpus dtype max_len log
  VLLM_CPU_OMP_THREADS_BIND=$2 LD_PRELOAD="$WORK/venv/lib/libiomp5.so" \
    venv/bin/vllm serve "$model" --served-model-name smollm2 --dtype "$3" \
    --max-model-len "$4" --enforce-eager --max-num-seqs 1 --host 127.0.0.1 \
    --port "$1" > "$5" 2>&1 &
  echo $!
}
wait_ready() {
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 2; done
  return 1
}

python3 - "$OUT/workload.jsonl" <<'PY'
import json, sys
rows = [("story", "Once upon a time"), ("dog", "The little dog"),
        ("capital", "The capital of France is"), ("arithmetic", "Two plus two equals")]
with open(sys.argv[1], "w") as f:
    for pid, prompt in rows:
        f.write(json.dumps({"id": pid, "prompt": prompt,
                            "checks": [{"id": "answered", "kind": "nonempty"}]}) + "\n")
PY
policy() {  # name agreement_min kld_max
  python3 - "$OUT/policy-$1.json" "$2" "$3" <<'PY'
import json, sys
json.dump({"schema": "metria.verification_policy.v1", "criteria": [
  {"target": "quality.candidate_pass_rate", "version": "1", "min": 1.0},
  {"target": "behavior.reference_repeatability", "version": "1", "min": 1.0},
  {"target": "behavior.trajectory_agreement", "version": "0.3.4", "min": float(sys.argv[2])},
  {"target": "behavior.trajectory_kld_mean", "version": "0.1.0", "max": float(sys.argv[3])},
]}, open(sys.argv[1], "w"), indent=2)
PY
}
policy strict 1.0 1e-9
policy bf16 0.9 0.1

ref_pid=$(serve 8801 0-1 float32 256 "$OUT/serve-reference.log")
wait_ready 8801 || { echo "reference server did not start"; exit 1; }
export METRIA_REFERENCE_URL=http://127.0.0.1:8801 METRIA_CANDIDATE_URL=http://127.0.0.1:8802

case_run() {  # name dtype max_len policy ref_declared... -- cand_declared...
  name=$1; dtype=$2; maxlen=$3; pol=$4; shift 4
  refdecl=(); canddecl=(); side=ref
  for item in "$@"; do
    if [ "$item" = "--" ]; then side=cand; continue; fi
    if [ $side = ref ]; then refdecl+=(--reference-declared "$item"); else canddecl+=(--candidate-declared "$item"); fi
  done
  cand_pid=$(serve 8802 2-3 "$dtype" "$maxlen" "$OUT/serve-$name.log")
  wait_ready 8802 || { echo "$name candidate did not start"; kill "$cand_pid"; return; }
  venv/bin/python -m metria recipe prepare-endpoint --model-id smollm2 \
    --tokenizer "$tok" --workload "$OUT/workload.jsonl" --policy "$OUT/policy-$pol.json" \
    --reference-url-env METRIA_REFERENCE_URL --candidate-url-env METRIA_CANDIDATE_URL \
    "${refdecl[@]}" "${canddecl[@]}" --max-tokens 16 --seed 42 \
    --warmup-trials 0 --measured-trials 2 --staged-kld --timeout 600 \
    --output "$OUT/$name.study.json" > "$OUT/$name.log" 2>&1
  venv/bin/python -m metria verify "$OUT/$name.study.json" --output "$OUT/$name" >> "$OUT/$name.log" 2>&1
  echo "$name exit=$?"
  kill "$cand_pid"; wait "$cand_pid" 2>/dev/null
}

case_run identical float32 256 strict dtype=float32 -- dtype=float32
case_run bf16 bfloat16 256 bf16 dtype=float32 -- dtype=bfloat16
case_run contradiction float32 256 strict engine_version=0.30.0 -- engine_version=0.29.0
case_run undeclared float32 512 strict dtype=float32 -- dtype=float32
kill "$ref_pid"; wait "$ref_pid" 2>/dev/null
echo QUALIFY_DONE
