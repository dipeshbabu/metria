# Installed-wheel llama.cpp build qualification

On 2026-09-28, the installed Metria wheel prepared and verified a controlled
CPU comparison between two distinct native capture providers. Both provider
qualification probes completed. The three-prompt comparison was valid and the
explicit completion-presence policy returned `PASS`.

- Metria source: `38438efc8072e1060e2182a7541ad1caf6509ba3`.
- Installed wheel SHA256: `632ae0a43eef24079fa90b9679d456bf797989b3e59487082a637b9b83ec13fc`.
- Native llama.cpp source: `434ddbbc0e30522e897670681e503b797c12b7c1` with
  capture patch SHA256 `75eb5872101ab933c192355c93acdc4126cb5e70f5f928aee04f555bb36f6997`.
- Reference provider: retained Release/O3 static build, SHA256
  `5e17486285ea5ecf5bce1bfd7bafa172fe3e0b0fea0ede0eafa59fe5c998c9b6`.
- Candidate provider: independently compiled Release/O2 static build using
  GCC 13.3, SHA256 `0fe6d5edf465771841c16dcdb52119a61c1a9d7300655f1ca8701dcd11141f42`.
- Same pinned stories260K GGUF, two generation threads, one batch thread,
  context 256, greedy plain completion, and three public story prefixes.

[qualification.json](qualification.json) records source, wheel, model and build
identities. [study.json](study.json), [workload.jsonl](workload.jsonl), and
[policy.json](policy.json) are the actual inputs. The independent probes are
[reference](study.reference.qualification.run.json) and
[candidate](study.candidate.qualification.run.json). The completed
[report](verification/report.md) and [canonical result](verification/verification.json)
retain both native run records. [files-sha256.json](files-sha256.json) binds the
17 original files, copied without rewriting their bytes.

These tiny-model runs establish the installed command, provider pinning,
comparison and policy contract. The checks require nonempty completions; they
do not judge correctness or usefulness. Cold-process timings include startup
and model loading and do not establish a statistical performance regression.
This is maintainer qualification, not external pilot feedback. Runtime/model
paths identify this retained environment; prepare new recipes for another host.
