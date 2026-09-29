"""Shared wire identity and role ordering for verification execution and reports."""

VERIFICATION_SCHEMA = "metria.verification.v1"
VERIFICATION_SCOPE = "local_llamacpp_cpu_threads.v1"
VERIFICATION_ROLES = ("reference", "candidate")

VLLM_VERIFICATION_SCOPE = "local_vllm_prefix_cache.v1"
LLAMACPP_BUILD_SCOPE = "local_llamacpp_cpu_builds.v1"
GGUF_QUANTIZATION_SCOPE = "local_llamacpp_gguf_quantization.v1"
