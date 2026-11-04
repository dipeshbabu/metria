"""Scoring extreme probabilities and bounding token chunk allocations."""

import math
from decimal import Decimal

import pytest

from metria.fidelity.measurement_math import (
    aggregate_topk_kld,
    approximate_topk_kl,
    full_token_chunks,
)


def test_chunk_limit_bounds_how_many_token_slices_are_copied():
    class Tokens(list):
        slices = 0

        def __getitem__(self, key):
            self.slices += 1
            return super().__getitem__(key)

    tokens = Tokens(range(100))
    assert full_token_chunks(tokens, chunk_len=5, max_chunks=2) == [
        list(range(5)),
        list(range(5, 10)),
    ]
    assert tokens.slices == 2


@pytest.mark.parametrize("candidate_logprob", [-740.0, -1000.0])
def test_topk_kl_retains_extreme_logprob_information(candidate_logprob):
    p = math.exp(-0.5)
    expected = p * (-0.5 - candidate_logprob) + (1 - p) * math.log1p(-p)
    reference, candidate = {1: -0.5}, {1: candidate_logprob}
    assert approximate_topk_kl(reference, candidate) == pytest.approx(expected)
    metrics = aggregate_topk_kld([[reference]], [[candidate]], backend_name="fixture")
    assert metrics.mean_kld == pytest.approx(expected)
    assert metrics.n_positions_scored == 1


def test_underflowing_floor_still_produces_finite_normalized_kl():
    assert approximate_topk_kl({1: 0.0}, {2: 0.0}, log_floor=-1000.0) == 1000.0
    assert approximate_topk_kl({1: -1000.0}, {1: -1000.0}) == 0.0


def test_topk_kl_preserves_support_for_numeric_logprob_types():
    assert approximate_topk_kl({1: Decimal("-0.5")}, {1: -0.5}) == 0.0


def test_omitted_mass_matches_analytic_binary_distribution_kl():
    assert approximate_topk_kl(
        {1: math.log(0.75)}, {1: math.log(0.25)}
    ) == pytest.approx(0.5 * math.log(3))
    assert approximate_topk_kl({1: -1000.0}, {1: -0.5}) == pytest.approx(
        -math.log1p(-math.exp(-0.5))
    )


@pytest.mark.parametrize("invalid", [True, "-1", None, -(10**400)])
def test_invalid_log_probabilities_raise_value_errors(invalid):
    with pytest.raises(ValueError, match="log probability"):
        approximate_topk_kl({1: invalid}, {1: -0.5})


@pytest.mark.parametrize("invalid", [True, "-30", None, -(10**400)])
def test_invalid_log_floor_raises_value_errors(invalid):
    with pytest.raises(ValueError, match="log_floor"):
        approximate_topk_kl({1: -0.5}, {1: -0.5}, log_floor=invalid)
