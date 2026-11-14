# Native CPU seeded sampling and local chat templates on vLLM endpoints

On 2026-11-13 `remote_endpoint_change.v1` ran against two identically
configured `vllm serve` processes from the pinned `0.30.0+cpu` wheel in WSL2
(SmolLM2-135M, float32, `--max-num-seqs 1`, separate CPU pairs) with production
decode settings:

- `--temperature 0.8 --seeds 1,2,3 --top-p 0.95 --stop "<|im_end|>"`;
- `--chat-template`. The pinned SmolLM2-135M base tokenizer ships no chat
  template, so for this check SmolLM2-Instruct's ChatML template was added to a
  copy of its `tokenizer_config.json` (pinned by hash in `study.json`). The
  template is a functional fixture, not a claim about the base model's chat
  quality.

The workload is the four prompts from [`endpoint-cpu`](../endpoint-cpu/README.md)
with a `nonempty` check, expanded to 12 seeded rows, with two measured trials.

| Check | Result |
| --- | --- |
| Verdict | **PASS** |
| Chat template identity | `verified`, rendered by Metria with jinja2 3.1.6 |
| Server `/tokenize` on the rendered prompts (`add_special_tokens=false`) | agreed on 4 of 4 |
| Seeds | 3 distinct samples per prompt; repeated trials identical per seed |
| Trajectory agreement between the two servers | 1.0: identical seeded samples |
| Candidate pass rate | 24 / 24, lower 95% Wilson bound 0.862 |
| Pass-rate difference | 0, lower 95% Newcombe bound −0.138 |

The interval width shows why seeded sampling needs enough prompts: even 24
passing checks only bound the candidate pass rate below by 0.86, and the checks
are correlated within a prompt.

Files: the unmodified verification bundle (`verification/`), the prepared
`study.json`, `policy.json` and `workload.jsonl`, and logs with local paths
replaced by placeholders.
