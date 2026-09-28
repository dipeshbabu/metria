from kv_fidelity import contracts, measurement_math
from kv_fidelity.backends import base


def test_backend_compatibility_names_use_shared_contracts_and_math():
    assert base.CompletionResult is contracts.CompletionResult
    assert base.TrajectoryResult is contracts.TrajectoryResult
    assert base.KLDResult is contracts.KLDResult
    assert base.BackendCapabilityError is contracts.BackendCapabilityError
    assert base._full_token_chunks is measurement_math.full_token_chunks
    assert base._aggregate_topk_kld is measurement_math.aggregate_topk_kld
    assert base.approximate_topk_kl is measurement_math.approximate_topk_kl
