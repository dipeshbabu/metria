"""Stable policy-facing identities for native serving measurements."""

from ..models import MetricDefinition
from .serving_metrics import DEFINITIONS as SOURCES

NAME = "metria.serving_impact"
VERSION = "1"
SCHEMA = "metria.serving_impact.v1"
DEFINITIONS = {
    name: MetricDefinition(
        name,
        definition.unit if suffix == "candidate" else "ratio",
        definition.direction,
        NAME,
        VERSION,
    )
    for key, definition in SOURCES.items()
    for suffix in ("candidate", "ratio")
    for name in (f"{key}_{suffix}",)
}
