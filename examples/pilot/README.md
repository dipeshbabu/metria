# Public maintainer pilot inputs

These authored tasks exercise literal output, sentiment classification, field
extraction and JSON structure. They contain no user data. Prompts use the chat
template from the immutable tokenizer pinned by
[model-manifest.json](model-manifest.json). The weights are not included.

Model: [HuggingFaceTB/SmolLM2-360M-Instruct at the pinned revision](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951),
Apache-2.0. The manifest covers the eight local model/tokenizer files used during
preparation. Verify downloaded bytes against it before use.

From an installed qualified vLLM/Metria environment, run:

```bash
metria recipe prepare-vllm \
  --model /path/to/model --descriptor model-manifest.json \
  --workload literal-contract.jsonl --policy policy.json \
  --context 512 --max-tokens 32 --warmup-trials 1 --measured-trials 2 \
  --output literal-study.json
metria verify literal-study.json --output literal-evidence
```

Repeat with `representative-tasks.jsonl` and a new output path. The predeclared
policy requires all reference and candidate task checks to pass. A separate
negative setup test uses that representative workload with `--max-tokens 1`;
truncated outputs must fail their actual task checks. Preserve its report.

Use [`metria pilot record`](../../docs/guides/verifier-pilot.md) to record setup
friction and decisions. These are
[maintainer tests](../../docs/guides/representative-maintainer-pilot.md), not
participant feedback or a production-quality model recommendation.
