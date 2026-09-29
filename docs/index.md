# Documentation

Metria verifies one inference change between a reference and a candidate. Start
with [getting started](guides/getting-started.md), the
[copyable workflows](../examples/verification/README.md), and
[CI integration](guides/verification-ci.md). The internal APIs support that path.

This directory contains current, maintained guidance for Metria. Dated
experiments and historical conclusions live under [`research/`](../research/README.md);
generated evidence lives under [`artifacts/`](../artifacts/README.md).

## Supporting architecture

- [Verifier product decision and scope](architecture/verifier-product.md)
- [Metria core architecture](architecture/metria-core.md)

## Guides

- [Task checks, repeatability and decision policies](guides/verification-decisions.md)
- [Installed-package workflow and synthetic demo](guides/installed-workflow.md)
- [Real-user pilot checklist](guides/verifier-pilot.md)
- [Unified fidelity methods and source migration](guides/unified-fidelity.md)

- [Verified model and data artifacts](guides/artifact-resolution.md)

- [Verify a local vLLM prefix-cache change](guides/vllm-prefix-verification.md)
- [FP8 KV-cache qualification status](guides/fp8-qualification-status.md)
- [Verify a local llama.cpp CPU thread change](guides/metria-verify.md)
- [Verify a llama.cpp CPU build change](guides/llamacpp-build-verification.md)
- [Verify a GGUF Q8_0 weight-quantization change](guides/gguf-quantization-verification.md)
- [Verify a pinned local vLLM CPU runtime upgrade](guides/runtime-upgrade-verification.md)
- [Copyable verification workflows and synthetic fixtures](../examples/verification/README.md)
- [Gate verification in CI](guides/verification-ci.md)
- [Metria llama.cpp runtime adapter](guides/metria-llamacpp-runtime.md)
- [Metria vLLM runtime adapter](guides/metria-vllm-runtime.md)
- [Metria trajectory measurement bridge](guides/metria-trajectory-measurement.md)
- [Metria pairwise analyses](guides/metria-pairwise-analysis.md)
- [TurboQuant configuration recommendations](guides/turboquant-recommendations.md)
- [MLX port](guides/mlx-port.md)
- [Windows and AMD RDNA 4 setup](guides/windows-rdna4-setup.md)

## Reference

- [Benchmarks](reference/benchmarks.md)
- [Hardware comparison matrix](reference/hardware-comparison-matrix.md)
- [Test-suite definition](reference/test-suite-definition.md)
- [Weight-compression results](reference/weight-compression-results.md)

## Historical engine setup

- [Archived TurboQuant+ fork setup](../research/archive/turboquant-engine-setup.md)

This archived guide describes an experimental fork whose public source is
unavailable. For the published Metria CLI, use the local verification guide above.

## Components

- [KV Fidelity](../components/kv-fidelity/README.md)
- [TurboQuant reference](../components/turboquant-reference/README.md)

## Maintainers

- [Project governance](../GOVERNANCE.md)
- [Maintainers and component ownership](../MAINTAINERS.md)
- [Repository settings baseline](maintainers/repository-settings.md)
- [Publishing the root Metria package](guides/releasing-metria.md)
- [Independent component release procedures](guides/releasing.md)

When documents disagree, prefer current guidance backed by the newer
controlled experiment. Preserve older results as dated evidence rather than
rewriting them to match the latest recommendation.
