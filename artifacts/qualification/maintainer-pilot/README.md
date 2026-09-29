# Installed-package maintainer pilot

These are automated maintainer results on public, authored tasks. No participant
feedback was collected. All three comparisons completed with valid native
identity/capture evidence and observed candidate prefix-cache reuse.

| Workload | Reference task pass rate | Candidate task pass rate | Exact token matches | Policy / decision |
| --- | --- | --- | --- | --- |
| Literal output contract | 100% | 100% | 100% | PASS / keep within this workload |
| Classification, extraction and JSON structure | 50% | 50% | 100% | FAIL / reject |
| Same tasks with a one-token budget | 16.7% | 16.7% | 100% | FAIL / reject |

The declared policy required every task check to pass for both roles. Perfect
token agreement did not make incorrect or truncated answers acceptable. The
identical reference/candidate pass rates do not show quality degradation caused
by prefix caching; they show the small model/setup failing the task contract.

Runtime: pinned vLLM `0.30.0+cu129` on the existing GTX 1650 (4 GiB).
Model: SmolLM2-360M-Instruct revision
`a10cc1512eabd3dde888204e902eca88bddb4951`, with eight immutable model/tokenizer
file pins. Each comparison used one warmup and two measured trials, context 512,
greedy generation and prefix caching off/on. The two normal cases used a 32-token
budget; the predeclared negative case used one token.

The executed source was `75844a8d0d37887c69a383d415f8b79832a6cf78`; installed wheel
SHA256 was `7c62bbaeab1d6efb6bb7eb947209ef9080e4029abce3c72f28ee72203dae9156`.
The source preceding that run rejected an administrative Hub tree-cache receipt
as model payload. `setup-failure/` retains the original failed preparation,
download metadata and source/wheel identities. The narrow metadata recognition
fix preserved every trusted model/tokenizer pin and all acceptance criteria.

Setup through saved verification took approximately 353, 357 and 317 seconds for
the three cases. Those times exclude the initial engine installation/model
download and the later pilot-record command. Windows component tests briefly ran
during the pilot and were stopped when host memory pressure rose. Host load was
not a controlled benchmark condition. Latency samples remain in the original
records, but these runs do not establish a production speedup or regression.

Each case preserves its inputs, preparation/validation/verification logs, full
run records, report, supplied notes and `metria pilot record` receipt. The content
index covers every retained file. Tests check hashes, native evidence, task
outcomes and replay of the current pilot-record binding. The receipts keep
`would_reuse` null and `participant_feedback_collected` remains false.

See the [workload description](../../../docs/guides/representative-maintainer-pilot.md)
and [participant protocol](../../../docs/guides/verifier-pilot.md). Actual user
feedback remains outstanding in [issue #159](https://github.com/dipeshbabu/metria/issues/159).
