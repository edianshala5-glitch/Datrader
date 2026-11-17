"""Indicator engine. Importing this package registers every indicator."""

from engine.indicators import context, smc, standard  # noqa: F401  (registration side effects)
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
