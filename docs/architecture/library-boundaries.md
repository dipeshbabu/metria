# Library boundaries and staged complexity reduction

Reusable operations live in Metria or the owning component, while scripts parse
arguments and call those operations. The public product remains one inference
change verification; extraction does not create a plugin ecosystem or a parallel
runtime/evaluator graph.

| Boundary | Reusable API | Compatibility |
|---|---|---|
| Verification rendering | `metria.reporting.render_verification` | The original `metria.verification` import remains an alias; canonical Markdown behavior is unchanged. |
| Verification wire identity | `metria.verification_schema` | Execution and rendering share schema/scope/role constants. |
| KV result/capability types | `kv_fidelity.contracts` | Backend-base imports remain available. |
| KLD chunking/alignment/validation/aggregation | `kv_fidelity.measurement_math` | Existing backend helper names delegate to the same functions; numerical methodology is unchanged. |
| Diagnostic platform policy and parsers | `metria.integrations.turboquant_diagnostics` | Historical script helpers remain available; policy eligibility is separate from observed hardware. |
| Historical diagnostic execution | `metria.integrations.turboquant_diagnostic_runner.run_diagnostic(DiagnosticOptions(...))` | The tool is a thin launcher; old imports resolve to the same implementation for existing callers/tests. |

Diagnostic integration policy stays explicitly TurboQuant-specific. It does not
become generic Metria capability policy. `platform_support_fingerprint()` maps
supplied platform facts while leaving accelerators unobserved. Native probes use
the shared bounded process runner. No model/runtime support is inferred from a
declared diagnostic platform policy.

## Targets and migration plan

Targets are 800 nonblank lines per module, 100 physical lines per function, and
Ruff C901 complexity at most 15. `check_complexity.py` enforces those targets for
new code and prevents growth above recorded ceilings in legacy code. The baseline
is a reviewed migration ledger, not permission to increase complexity by refreshing
it whenever a check fails.

This stage extracts complete reusable boundaries and preserves behavior with
existing renderer, backend, diagnostic, and CLI characterization suites. The
diagnostic launcher delegates to the library; reusable code no longer accumulates
under its script path. Its legacy section renderer remains large inside the
explicit integration, with a fixed no-growth ceiling.

Next stages reduce those recorded exceptions by responsibility: split diagnostic
inventory/monitoring/section rendering; move remaining KV argument-to-operation
translation out of its large CLI; and keep report templates separate from numerical
aggregation. Each stage must preserve characterized output or document a deliberate
behavior correction. Lower/remove the corresponding baseline entry as debt is
removed. Runtime/evaluator gates continue to apply to the new owning modules.

Benchmark lifecycle work uses the existing execution, process, evidence, and
verification APIs tracked in #17/#14. It must not establish another service/runtime
architecture under `tools/`. Engine-specific mechanics remain in adapters;
method-specific KLD math remains in KV Fidelity; root Metria owns common lifecycle,
identity, comparison, and verification contracts.
