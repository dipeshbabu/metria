# Evidence-focused test confidence

The aggregate component coverage floor remains 86%. Additional independent
module floors in `tools/maintenance/coverage-gates.json` prevent well-covered
code from hiding a weak runtime, evaluator, comparison guard, or serializer.
Core gates run in the existing Linux/Windows/macOS matrix; component gates run
in the existing Ubuntu coverage job. The aggregate component gate is unchanged.

The checker uses coverage.py's line-plus-branch counts, requires branch data,
fails on missing/ambiguous modules, and reports each module separately. The
core measurement command disables the unrelated component-wide default floor;
its next required step enforces the individual core floors. It does not remove
any component requirement.

Initial floors are ratchets below observed baselines, not quality certifications.
The optional KV Fidelity vLLM adapter now has offline execution/cache/tokenization
contract tests, measured about 97% coverage, and its own 95% gate. SGLang retains a separate lower baseline
that must improve independently; other modules cannot compensate for it.

Generated-input property tests cover trajectory score bounds/reflexivity/symmetry,
chunk order/alignment, canonical digest determinism, and finite serialization.
Fault-injection tests corrupt identity, lifecycle, method, missing evidence,
and provenance fields. Existing policy/provenance tests retain their own guard
fault cases. This is equivalent fault testing of important rejection paths,
not a claim of a repository-wide mutation-testing score.

Required retained qualification fixtures and the bundled prompt corpus are checked
at test-session start. Their disappearance fails rather than silently skipping
the integration evidence. CI uploads machine-readable coverage, per-module
Markdown summaries, and skipped test IDs/reasons. Expected platform/optional
integration skips remain visible for review.

Two small subprocess entry points have explicit threshold exclusions:

- `__main__.py` delegates to the covered CLI and is exercised by subprocess tests.
- `_process_windows.py` is an isolated native child bootstrap. Parent-process
  coverage does not measure it; real Windows process-tree tests check timeout,
  success, callback failure, interruption, and cleanup. Its zero in the JSON report
  is retained, not hidden or presented as measured coverage.

Export-only initializers are measured/reported without separate scientific-guard
floors. Optional runtime/evaluator modules remain included. Legacy research tools
outside the package sources keep their focused contract tests; their execution
is not included in a misleading package-wide denominator.

These tests establish implementation confidence. They do not replace the pinned
[real-engine qualification](runtime-qualification.md), and mocked adapter coverage
does not expand the supported runtime/model/hardware scope.
