"""STAGED verification profile for two running OpenAI-compatible endpoints.

``remote_endpoint_change.v1`` compares a reference and a candidate server with
the same pinned tokenizer, workload, decode settings and task checks. Server
identity is only *claimed* (see ``runtimes.endpoint_identity``), so the profile
always requires an explicit policy and ends in PASS or FAIL, never a bare
VERIFIED. The intended change is stated in each run's ``runtime.declared``
block; ``comparison.vary`` must equal the set this module derives from it, so
any other server-reported difference makes the runs not comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from .measurements import TokenTrajectoryProtocol, TrajectoryDivergenceAnalysis
from .measurements import batch_invariance as invariance
from .measurements.endpoint_sampling import expand_seeds, validate_scenario
from .measurements.prefix_distributions import DISTRIBUTIONS_KEY, staged_distributions
from .measurements.prefix_workload import (
    WORKLOAD_METHOD,
    PrefixWorkloadProtocol,
    trajectory_config,
    trial_settings,
)
from .measurements.trajectory_analysis import TrajectoryAgreementAnalysis
from .measurements.verification_impact import VerificationImpactAnalysis
from .models import RunRecord, RunSpec
from .recipes import StudyRecipe
from .runtimes.endpoint import RUNTIME_NAME, TOKEN_FLAVORS, EndpointAdapter
from .runtimes.endpoint_identity import identity_gaps, validate_declared
from .verification_evidence import task_check_gaps, task_check_identity
from .verification_route import VerificationRoute
from .verification_schema import ENDPOINT_SCOPE

ISOLATION = "none: remote server state is not reset between trials"
_CLAIMED = {
    "engine_version": ("observed.identity.runtime", "runtime"),
    "max_model_len": ("observed.identity.model", "model"),
    "model_root": ("observed.identity.model", "model"),
}


class EndpointWorkloadProtocol(PrefixWorkloadProtocol):
    """Sequential trials against a server Metria cannot reset.

    With a ``concurrency`` block, a concurrent pass follows the sequential
    trials and its batch-invariance evidence joins the trajectory evidence.
    """

    reset_scope = None
    isolation = ISOLATION

    def requirements(self, config: Mapping[str, Any]) -> tuple[Any, ...]:
        return super().requirements(invariance.without_concurrency(config))

    def execute(
        self, session: Any, scenario: Mapping[str, Any], config: Mapping[str, Any]
    ) -> Any:
        from .protocols import MeasurementResult

        config = expand_seeds(config, scenario)
        settings = invariance.concurrency_settings(config)
        result = super().execute(
            session, scenario, invariance.without_concurrency(config)
        )
        if settings is None:
            return result
        evidence = dict(result.evidence)
        evidence[invariance.EVIDENCE_KEY] = invariance.measure(
            session, config, settings, result.evidence
        )
        return MeasurementResult(
            metrics=result.metrics, evidence=evidence, artifacts=result.artifacts
        )


def is_endpoint_recipe(recipe: StudyRecipe) -> bool:
    return any(run.runtime.get("name") == RUNTIME_NAME for run in recipe.study.runs)


def trial_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    from .integrations.staged_captures import STAGED_CAPTURE_CHECK

    warmup, measured = trial_settings(config)
    identity: dict[str, Any] = {
        "method": WORKLOAD_METHOD,
        "warmup_trials": warmup,
        "measured_trials": measured,
        "schedule": "sequential",
        "isolation": "none_remote_server",
    }
    if staged_distributions(config):
        identity["capability_overrides"] = (STAGED_CAPTURE_CHECK,)
    settings = invariance.concurrency_settings(config)
    if settings is not None:
        identity["concurrency"] = settings
    return identity


def analyses(config: Mapping[str, Any]) -> tuple[str, ...]:
    names: tuple[str, ...] = (
        TrajectoryAgreementAnalysis.name,
        VerificationImpactAnalysis.name,
    )
    if staged_distributions(config):
        names += (TrajectoryDivergenceAnalysis.name,)
    if invariance.concurrency_settings(config) is not None:
        names += (invariance.NAME,)
    return names


def derived_vary(left: RunSpec, right: RunSpec) -> frozenset[str]:
    """Endpoint plumbing plus every path the declared change may move."""

    paths = {
        "runtime.base_url_env",
        "resolved.runtime.base_url_env",
        "observed.endpoint",
        "observed.identity.endpoint",
    }
    if "api_key_env" in left.runtime and "api_key_env" in right.runtime:
        paths |= {"runtime.api_key_env", "resolved.runtime.api_key_env"}
    declared = [validate_declared(run.runtime.get("declared")) for run in (left, right)]
    for key in sorted(set(declared[0]) | set(declared[1])):
        if declared[0].get(key) == declared[1].get(key):
            continue
        if key not in declared[0] or key not in declared[1]:
            raise ValueError(f"runtime.declared.{key} must be stated for both runs")
        paths |= {
            f"runtime.declared.{key}",
            f"resolved.declared.{key}",
            f"observed.declared.{key}",
        }
        if key in _CLAIMED:
            component, snapshot = _CLAIMED[key]
            paths |= {
                component,
                f"observed.identity_snapshots.before.{snapshot}",
                f"observed.identity_snapshots.after.{snapshot}",
            }
    return frozenset(paths)


def _validate_runs(left: RunSpec, right: RunSpec, config: Mapping[str, Any]) -> None:
    for run in (left, right):
        if run.runtime.get("name") != RUNTIME_NAME:
            raise ValueError("endpoint verification requires two endpoint runs")
        if run.runtime.get("flavor") not in TOKEN_FLAVORS:
            raise ValueError(
                "endpoint verification requires a flavor that returns token IDs"
            )
        if run.treatments or run.environment_selector:
            raise ValueError("endpoint runs do not accept treatments or placement")
        if run.trial_policy != trial_identity(config):
            raise ValueError("endpoint run trial policy does not match the workload")
        if run.measurements != (TokenTrajectoryProtocol.name,):
            raise ValueError("endpoint runs measure exactly the trajectory workload")
        validate_scenario(run.scenario)
        if run.scenario.get("chat_template") and "tokenizer_config.json" not in (
            run.model.get("tokenizer_files", {})
        ):
            raise ValueError(
                "scenario.chat_template requires a pinned tokenizer_config.json"
            )
    if left.model != right.model or left.scenario != right.scenario:
        raise ValueError(
            "endpoint verification requires identical model and scenario inputs"
        )
    plumbing = {"base_url_env", "api_key_env", "declared"}
    if {k: v for k, v in left.runtime.items() if k not in plumbing} != {
        k: v for k, v in right.runtime.items() if k not in plumbing
    }:
        raise ValueError("endpoint runs differ outside their endpoint and declaration")
    if left.runtime["base_url_env"] == right.runtime["base_url_env"]:
        raise ValueError("reference and candidate must name different URL variables")


def validate_endpoint_recipe(recipe: StudyRecipe) -> dict[str, Any]:
    """Validate the profile; return the reference and candidate declarations."""

    if len(recipe.study.runs) != 2:
        raise ValueError("verify requires exactly two runs: reference, then candidate")
    if set(recipe.measurement_configs) != {TokenTrajectoryProtocol.name}:
        raise ValueError("endpoint verification requires one trajectory workload")
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    trajectory_config(invariance.without_concurrency(config))
    invariance.concurrency_settings(config)
    if recipe.policy is None:
        raise ValueError("endpoint verification requires an explicit acceptance policy")
    if task_check_identity(config) is None:
        raise ValueError("endpoint verification requires declared task checks")
    if recipe.environment:
        raise ValueError(
            "endpoint verification reads URLs and keys from named environment "
            "variables only; recipe.environment must be empty"
        )
    left, right = recipe.study.runs
    _validate_runs(left, right, config)
    comparison = recipe.study.comparison
    if comparison.analyses != analyses(config):
        raise ValueError(
            "endpoint verification requires trajectory and impact analyses, plus "
            "metria.trajectory_kld exactly when staged_distributions is true, "
            "and metria.batch_invariance exactly when concurrency is declared"
        )
    if comparison.waivers or comparison.vary != derived_vary(left, right):
        raise ValueError(
            "comparison.vary must equal the paths derived from runtime.declared; "
            "use metria recipe prepare-endpoint"
        )
    return {
        "reference": validate_declared(left.runtime.get("declared")),
        "candidate": validate_declared(right.runtime.get("declared")),
    }


def _workload_gaps(record: RunRecord, trials: Mapping[str, Any]) -> list[str]:
    measurements = record.evidence.get("measurements")
    capture = (
        measurements.get(TokenTrajectoryProtocol.name)
        if isinstance(measurements, Mapping)
        else None
    )
    workload = capture.get("workload") if isinstance(capture, Mapping) else None
    if (
        not isinstance(workload, Mapping)
        or workload.get("method") != WORKLOAD_METHOD
        or workload.get("isolation") != ISOLATION
        or any(
            workload.get(key) != trials[key]
            for key in ("warmup_trials", "measured_trials")
        )
    ):
        return ["declared endpoint trial evidence is missing or inconsistent"]
    samples = workload.get("samples")
    invocations = record.observed.get("invocations")
    if (
        not isinstance(samples, Sequence)
        or not isinstance(invocations, Sequence)
        or not samples
        or len(samples) != len(invocations)
    ):
        return ["endpoint request and workload sample counts differ or are missing"]
    for sample, invocation in zip(samples, invocations, strict=True):
        if (
            not isinstance(invocation, Mapping)
            or not isinstance(invocation.get("completion_tokens"), int)
            or invocation.get("prompt_tokens") != sample.get("prompt_tokens")
        ):
            return ["endpoint request evidence is missing or inconsistent"]
    return []


def _distribution_gaps(record: RunRecord) -> list[str]:
    capture = record.evidence.get("measurements", {}).get(
        TokenTrajectoryProtocol.name, {}
    )
    retained = capture.get(DISTRIBUTIONS_KEY) if isinstance(capture, Mapping) else None
    if not isinstance(retained, Mapping) or retained.get("n_prompts") != capture.get(
        "n_prompts"
    ):
        return ["staged distribution evidence is missing or incomplete"]
    return []


def _concurrency_gaps(record: RunRecord, settings: Mapping[str, int]) -> list[str]:
    measurements = record.evidence.get("measurements")
    capture = (
        measurements.get(TokenTrajectoryProtocol.name)
        if isinstance(measurements, Mapping)
        else None
    )
    capture = capture if isinstance(capture, Mapping) else {}
    retained = capture.get(invariance.EVIDENCE_KEY)
    workload = capture.get("workload")
    prompts = capture.get("n_prompts")
    trials = workload.get("measured_trials") if isinstance(workload, Mapping) else None
    concurrent = record.observed.get("concurrent_invocations", ())
    if (
        not isinstance(retained, Mapping)
        or not isinstance(prompts, int)
        or not isinstance(trials, int)
        or trials < 1
        or retained.get("max_in_flight") != settings["max_in_flight"]
        or retained.get("requests") != prompts // trials * settings["rounds"]
        or len(concurrent) != retained.get("requests")
    ):
        return ["declared concurrent schedule evidence is missing or incomplete"]
    return []


def evidence_gaps(
    record: RunRecord,
    *,
    trials: Mapping[str, Any],
    quality_identity: Mapping[str, Any] | None,
    distributions: bool,
    concurrency: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    gaps = [
        *identity_gaps(record.observed),
        *_workload_gaps(record, trials),
        *task_check_gaps(record, quality_identity),
    ]
    if distributions:
        gaps += _distribution_gaps(record)
    if concurrency is not None:
        gaps += _concurrency_gaps(record, concurrency)
    return tuple(dict.fromkeys(gaps))


def _section(value: Any, key: str) -> Mapping[str, Any]:
    item = value.get(key) if isinstance(value, Mapping) else None
    return item if isinstance(item, Mapping) else {}


def observed_facts(record: RunRecord) -> dict[str, Any]:
    identity = _section(record.observed, "identity")
    runtime = _section(identity, "runtime")
    model = _section(identity, "model")
    return {
        "identity_authority": record.observed.get("identity_authority"),
        "identity_status": identity.get("status"),
        "claimed_runtime_version": runtime.get("version"),
        "claimed_model": model.get("id"),
        "context": model.get("max_model_len"),
        "declared": dict(_section(record.observed, "declared")),
    }


def endpoint_performance(
    reference: RunRecord, candidate: RunRecord, comparable: bool
) -> dict[str, Any]:
    from .measurements.performance import compare_latency_results
    from .measurements.prefix_performance import DEFINITION, measure_prefix_performance

    result = compare_latency_results(
        measure_prefix_performance(reference),
        measure_prefix_performance(candidate),
        comparable=comparable,
        definition=DEFINITION,
        limitations="Client-side wall time per sequential request, including the network, load balancer and any concurrent traffic on the server. Descriptive only; not a policy target.",
    )
    return {**result, "label": "Client-side request latency (includes network)"}


def build_route(recipe: StudyRecipe) -> VerificationRoute:
    declarations = validate_endpoint_recipe(recipe)
    measurement = EndpointWorkloadProtocol()
    config = recipe.measurement_configs[TokenTrajectoryProtocol.name]
    trials = trial_identity(config)
    distributions = staged_distributions(config)
    registered: dict[str, Any] = {
        TrajectoryAgreementAnalysis.name: TrajectoryAgreementAnalysis(),
        VerificationImpactAnalysis.name: VerificationImpactAnalysis(),
    }
    if distributions:
        registered[TrajectoryDivergenceAnalysis.name] = TrajectoryDivergenceAnalysis()
    concurrency = invariance.concurrency_settings(config)
    if concurrency is not None:
        registered[invariance.NAME] = invariance.BatchInvarianceAnalysis()
    return VerificationRoute(
        scope=ENDPOINT_SCOPE,
        adapters={RUNTIME_NAME: EndpointAdapter()},
        measurements={measurement.name: measurement},
        analyses=registered,
        evidence_gaps=partial(
            evidence_gaps,
            trials=trials,
            quality_identity=task_check_identity(
                expand_seeds(config, recipe.study.runs[0].scenario),
                trials["measured_trials"],
            ),
            distributions=distributions,
            concurrency=concurrency,
        ),
        observed_facts=observed_facts,
        change={
            "parameter": "declared_endpoint_configuration",
            **declarations,
            "identity_authority": "claimed",
        },
        performance=endpoint_performance,
    )
