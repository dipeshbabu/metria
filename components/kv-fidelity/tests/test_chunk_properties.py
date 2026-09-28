from __future__ import annotations

import pytest

from kv_fidelity.backends.base import (
    BackendCapabilityError,
    _aggregate_topk_kld,
    _full_token_chunks,
)


def test_chunking_preserves_order_and_never_scores_partial_chunks():
    for count in range(35):
        tokens = list(range(count))
        for length in range(1, 10):
            for limit in (1, 2, 7):
                chunks = _full_token_chunks(tokens, chunk_len=length, max_chunks=limit)
                assert len(chunks) <= limit
                assert all(len(chunk) == length for chunk in chunks)
                flattened = [token for chunk in chunks for token in chunk]
                assert flattened == tokens[: len(flattened)]
                if len(chunks) < limit:
                    assert count - len(flattened) < length


def test_alignment_faults_are_not_silently_truncated():
    distribution = {1: -0.7, 2: -0.7}
    for reference, candidate in (
        ([[distribution, distribution]], [[distribution]]),
        ([[distribution]], [[distribution], [distribution]]),
    ):
        with pytest.raises(BackendCapabilityError):
            _aggregate_topk_kld(reference, candidate, backend_name="fixture")
