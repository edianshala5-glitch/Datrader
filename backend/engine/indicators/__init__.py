"""Indicator engine. Importing this package registers every indicator."""

from engine.indicators import (  # noqa: F401  (registration side effects)
    context,
    smc,
    standard,
    zones,
)
from engine.indicators.base import (
    REGISTRY,
    Category,
    FeatureRef,
    IndicatorSpec,
    Inputs,
    OutputKind,
    compute,
    get_spec,
)

__all__ = [
    "REGISTRY",
    "Category",
    "FeatureRef",
    "IndicatorSpec",
    "Inputs",
    "OutputKind",
    "compute",
    "get_spec",
]
