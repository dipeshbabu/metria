"""Method-bound streaming, serving throughput and native allocator measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import MetricDefinition, MetricDirection, MetricSample, MetricSummary
from ..protocols import MeasurementResult

DEFINITIONS = {
    "request_latency_seconds": MetricDefinition(
        "request_latency_seconds",
        "seconds",
        MetricDirection.LOWER_IS_BETTER,
        "metria.vllm_serving_request_latency",
        "1",
    ),
    "ttft_seconds": MetricDefinition(
        "ttft_seconds",
        "seconds",
        MetricDirection.LOWER_IS_BETTER,
        "metria.vllm_stream_first_token_latency",
        "1",
    ),
    "inter_token_latency_seconds": MetricDefinition(
        "inter_token_latency_seconds",
        "seconds",
        MetricDirection.LOWER_IS_BETTER,
        "metria.vllm_stream_inter_token_latency",
        "1",
    ),
    "decode_tokens_per_second": MetricDefinition(
        "decode_tokens_per_second",
        "tokens/second",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.vllm_stream_decode_rate",
        "1",
    ),
    "output_tokens_per_second": MetricDefinition(
        "output_tokens_per_second",
        "tokens/second",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.vllm_serving_output_throughput",
        "1",
    ),
    "requests_per_second": MetricDefinition(
        "requests_per_second",
        "requests/second",
        MetricDirection.HIGHER_IS_BETTER,
        "metria.vllm_serving_request_throughput",
        "1",
    ),
    "peak_device_allocated_bytes": MetricDefinition(
        "peak_device_allocated_bytes",
        "bytes",
        MetricDirection.LOWER_IS_BETTER,
        "metria.vllm_torch_peak_allocated",
        "1",
    ),
    "peak_device_reserved_bytes": MetricDefinition(
        "peak_device_reserved_bytes",
        "bytes",
        MetricDirection.LOWER_IS_BETTER,
        "metria.vllm_torch_peak_reserved",
        "1",
    ),
    "allocated_kv_bytes": MetricDefinition(
        "allocated_kv_bytes",
        "bytes",
        MetricDirection.LOWER_IS_BETTER,
        "metria.vllm_kv_tensor_storage",
        "1",
    ),
}


def _number(value: Any, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            "native serving measurements require finite nonnegative values"
        )
    return float(value)


def _request(value: Any) -> tuple[float, float, list[float] | None, int]:
    if not isinstance(value, Mapping) or value.get("finished") is not True:
        raise ValueError("native serving request is incomplete")
    latency = _number(value.get("request_latency_seconds"), positive=True)
    tokens = value.get("output_tokens")
    chunks = value.get("chunks")
    if (
        type(tokens) is not int
        or tokens <= 0
        or not isinstance(chunks, (list, tuple))
        or not chunks
    ):
        raise ValueError("native serving token/chunk evidence is missing")
    offsets = []
    counts = []
    for row in chunks:
        if (
            not isinstance(row, Mapping)
            or type(row.get("tokens")) is not int
            or row["tokens"] <= 0
        ):
            raise ValueError("native stream chunks require positive token counts")
        offsets.append(_number(row.get("offset_seconds")))
        counts.append(row["tokens"])
    if offsets != sorted(offsets) or offsets[-1] > latency or sum(counts) != tokens:
        raise ValueError("stream timing and token capture evidence is inconsistent")
    gaps = (
        [right - left for left, right in zip(offsets, offsets[1:], strict=False)]
        if len(counts) > 1 and all(count == 1 for count in counts)
        else None
    )
    return latency, offsets[0], gaps, tokens


def _summary(
    name: str,
    values: Sequence[float],
    *,
    aggregation: str = "mean",
    total_value: float | None = None,
) -> MetricSummary:
    result = (
        math.fsum(value / len(values) for value in values)
        if aggregation == "mean"
        else max(values)
    )
    return MetricSummary(
        definition=DEFINITIONS[name],
        value=result if total_value is None else total_value,
        samples=tuple(
            MetricSample(value, metadata={"sample": index})
            for index, value in enumerate(values)
        ),
        aggregation=aggregation,
        coverage=1.0,
    )


def _memory(rows: Sequence[Mapping[str, Any]]) -> dict[str, MetricSummary]:
    if not rows or any(row.get("available") is not True for row in rows):
        return {}
    if any(
        row.get("method") != "metria.vllm_torch_allocator_and_kv_storage"
        or row.get("version") != "1"
        or row.get("reset", {}).get("reset") != "torch_cuda_allocator_peak"
        for row in rows
    ):
        raise ValueError("native memory method identity is incompatible")
    names = {
        "peak_device_allocated_bytes": "peak_allocated_bytes",
        "peak_device_reserved_bytes": "peak_reserved_bytes",
        "allocated_kv_bytes": "allocated_kv_bytes",
    }
    metrics = {}
    for name, field in names.items():
        values = []
        for row in rows:
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    "native memory measurements must be nonnegative byte counts"
                )
            values.append(float(value))
        metrics[name] = _summary(name, values, aggregation="max")
    return metrics


def measure_serving(
    batches: Sequence[Mapping[str, Any]],
    memories: Sequence[Mapping[str, Any]],
    *,
    prompts_per_trial: int,
) -> MeasurementResult:
    if (
        type(prompts_per_trial) is not int
        or prompts_per_trial < 1
        or not batches
        or len(memories) != len(batches)
    ):
        raise ValueError(
            "serving trials and native memory observations must be complete"
        )
    latencies: list[float] = []
    first_tokens: list[float] = []
    intervals: list[float] = []
    complete_intervals = True
    durations = []
    counts = []
    for batch in batches:
        if (
            batch.get("schema") != "metria.vllm_serving_batch.v1"
            or batch.get("clock") != "perf_counter_ns"
            or batch.get("transport") != "local AsyncLLM.generate"
        ):
            raise ValueError("serving timing source is missing or incompatible")
        requests = batch.get("requests")
        if (
            not isinstance(requests, (list, tuple))
            or len(requests) != prompts_per_trial
        ):
            raise ValueError("serving trial did not retain every request")
        duration = _number(batch.get("window_seconds"), positive=True)
        tokens = 0
        for request in requests:
            latency, first, gaps, count = _request(request)
            if latency > duration:
                raise ValueError("request latency exceeds the measured serving window")
            latencies.append(latency)
            first_tokens.append(first)
            complete_intervals &= gaps is not None
            intervals.extend(gaps or [])
            tokens += count
        durations.append(duration)
        counts.append(tokens)
    metrics = {
        "request_latency_seconds": _summary("request_latency_seconds", latencies),
        "ttft_seconds": _summary("ttft_seconds", first_tokens),
    }
    if complete_intervals and intervals:
        metrics["inter_token_latency_seconds"] = _summary(
            "inter_token_latency_seconds", intervals
        )
        if math.fsum(intervals) > 0:
            metrics["decode_tokens_per_second"] = _summary(
                "decode_tokens_per_second",
                [len(intervals) / math.fsum(intervals)],
                aggregation="ratio_of_totals",
                total_value=len(intervals) / math.fsum(intervals),
            )
    metrics["output_tokens_per_second"] = _summary(
        "output_tokens_per_second",
        [count / duration for count, duration in zip(counts, durations, strict=True)],
        aggregation="ratio_of_totals",
        total_value=sum(counts) / math.fsum(durations),
    )
    metrics["requests_per_second"] = _summary(
        "requests_per_second",
        [prompts_per_trial / duration for duration in durations],
        aggregation="ratio_of_totals",
        total_value=prompts_per_trial * len(batches) / math.fsum(durations),
    )
    metrics.update(_memory(memories))
    return MeasurementResult(
        metrics=metrics,
        evidence={
            "schema": "metria.serving_measurements.v1",
            "available": True,
            "unavailable": {
                name: "native individual-token intervals or CUDA allocation evidence are unavailable"
                for name in DEFINITIONS
                if name not in metrics
            },
            "boundaries": {
                "transport": "local native serving API; excludes HTTP/network overhead",
                "request_start": "admitted client request submitted to AsyncLLM.generate",
                "first_token": "first streamed chunk containing native generated tokens",
                "inter_token": "only when every observed chunk contains exactly one token",
                "throughput": "completed requests and native output tokens divided by whole measured batch windows",
                "memory": "native worker PyTorch allocator peaks and unique KV tensor storage; not device-wide memory usage",
            },
        },
    )
