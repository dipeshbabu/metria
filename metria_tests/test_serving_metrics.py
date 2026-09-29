from copy import deepcopy

import pytest

from metria.measurements.serving_metrics import measure_serving


def batch(window=4.0, chunks=None):
    return {
        "schema": "metria.vllm_serving_batch.v1",
        "clock": "perf_counter_ns",
        "transport": "local AsyncLLM.generate",
        "concurrency_limit": 2,
        "peak_inflight": 2,
        "window_seconds": window,
        "requests": [
            {
                "finished": True,
                "request_latency_seconds": 3.0,
                "output_tokens": 3,
                "chunks": chunks
                or [
                    {"offset_seconds": 1.0, "tokens": 1},
                    {"offset_seconds": 1.5, "tokens": 1},
                    {"offset_seconds": 2.0, "tokens": 1},
                ],
            }
        ],
    }


def memory():
    return {
        "available": True,
        "method": "metria.vllm_torch_allocator_and_kv_storage",
        "version": "1",
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "allocated_kv_bytes": 60,
        "reset": {"available": True, "reset": "torch_cuda_allocator_peak"},
    }


def test_streaming_rates_use_native_tokens_and_ratio_of_whole_windows():
    result = measure_serving(
        [batch(4), batch(8)],
        [memory(), {**memory(), "peak_allocated_bytes": 150}],
        prompts_per_trial=1,
    )
    assert result.metrics["output_tokens_per_second"].value == 0.5
    assert result.metrics["requests_per_second"].value == pytest.approx(2 / 12)
    assert result.metrics["ttft_seconds"].value == 1
    assert result.metrics["inter_token_latency_seconds"].value == 0.5
    assert result.metrics["decode_tokens_per_second"].value == 2
    assert result.metrics["peak_device_allocated_bytes"].value == 150
    assert result.metrics["allocated_kv_bytes"].value == 60


def test_coalesced_chunks_do_not_invent_individual_token_intervals():
    result = measure_serving(
        [batch(chunks=[{"offset_seconds": 2, "tokens": 3}])],
        [{"available": False}],
        prompts_per_trial=1,
    )
    assert result.metrics["ttft_seconds"].value == 2
    assert "inter_token_latency_seconds" not in result.metrics
    assert "decode_tokens_per_second" not in result.metrics
    assert "allocated_kv_bytes" not in result.metrics
    assert set(result.evidence["unavailable"]) >= {
        "decode_tokens_per_second",
        "allocated_kv_bytes",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("clock", "time.time"),
        ("transport", "http"),
        ("schema", "other"),
        ("window_seconds", 0),
        ("window_seconds", float("nan")),
        ("requests", []),
        ("window_seconds", 2),
    ],
)
def test_incomplete_or_incompatible_windows_are_rejected(field, value):
    with pytest.raises(ValueError):
        measure_serving([{**batch(), field: value}], [memory()], prompts_per_trial=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("finished", False),
        ("output_tokens", True),
        ("output_tokens", 4),
        ("chunks", []),
        ("chunks", [{"offset_seconds": 4, "tokens": 3}]),
        (
            "chunks",
            [{"offset_seconds": 2, "tokens": 1}, {"offset_seconds": 1, "tokens": 2}],
        ),
        ("chunks", [{"offset_seconds": 1, "tokens": False}]),
        ("request_latency_seconds", -1),
    ],
)
def test_invalid_native_stream_never_becomes_measurement(field, value):
    data = deepcopy(batch())
    data["requests"][0][field] = value
    with pytest.raises(ValueError):
        measure_serving([data], [memory()], prompts_per_trial=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "other"),
        ("version", "2"),
        ("peak_allocated_bytes", -1),
        ("peak_reserved_bytes", True),
    ],
)
def test_invalid_memory_identity_or_counts_are_rejected(field, value):
    with pytest.raises(ValueError):
        measure_serving([batch()], [{**memory(), field: value}], prompts_per_trial=1)


def test_partial_memory_coverage_does_not_report_a_peak():
    result = measure_serving(
        [batch(), batch()], [memory(), {"available": False}], prompts_per_trial=1
    )
    assert "peak_device_allocated_bytes" not in result.metrics


@pytest.mark.parametrize(
    "batches,memories,count",
    [([], [], 1), ([batch()], [], 1), ([batch()], [memory()], 0)],
)
def test_missing_measurement_series_is_rejected(batches, memories, count):
    with pytest.raises(ValueError):
        measure_serving(batches, memories, prompts_per_trial=count)
