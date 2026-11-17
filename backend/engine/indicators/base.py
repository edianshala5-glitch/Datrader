"""Indicator registry: typed parameters, typed outputs, shared inputs.

Every indicator is a pure function ``fn(inputs, **params) -> {output: array}`` registered with:

* **params** with ranges. The genetic algorithm (Step 6) mutates within them.
* **outputs** with a kind. The strategy DSL (Step 4) uses kinds to form only meaningful
  conditions: compare a ``price`` output with the close, threshold an ``oscillator`` within
  its range, test an ``event``/``flag`` for truth, match a ``state`` against its values.

No-lookahead contract: the value at bar ``t`` may depend only on bars ``<= t``, and is
known at the *close* of bar ``t``. ``tests/unit/test_indicators_lookahead.py`` enforces this
for every registered indicator by comparing full-history and truncated computations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar

FloatArray = npt.NDArray[np.float64]
Outputs = dict[str, npt.NDArray[Any]]


class Category(StrEnum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    STRUCTURE = "structure"
    LIQUIDITY = "liquidity"
    DIVERGENCE = "divergence"
    REGIME = "regime"
    CONTEXT = "context"


class OutputKind(StrEnum):
    PRICE = "price"  # same units as price, so comparable with close or other price outputs
    OSCILLATOR = "oscillator"  # thresholded within [lo, hi]
    UNBOUNDED = "unbounded"  # thresholded, scale depends on market (e.g. MACD)
    EVENT = "event"  # bool, True on the bar something happened
    FLAG = "flag"  # bool, persistent condition
    STATE = "state"  # small integer category (trend -1/0/1, weekday 0..4)
    RATIO = "ratio"  # dimensionless (ATR multiples, percent rank 0..1)


@dataclass(frozen=True)
class Param:
    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None

    @property
    def kind(self) -> str:
        if self.choices is not None:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        return "int" if isinstance(self.default, int) else "float"

    def validate(self, name: str, value: Any) -> Any:
        kind = self.kind
        if kind == "choice":
            if value not in self.choices:  # type: ignore[operator]
                raise ValueError(f"{name}={value!r} not in {self.choices}")
            return value
        if kind == "bool":
            return bool(value)
        coerced: int | float = int(value) if kind == "int" else float(value)
        if kind == "int" and coerced != value:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if (self.low is not None and coerced < self.low) or (
            self.high is not None and coerced > self.high
        ):
            raise ValueError(f"{name}={coerced} outside [{self.low}, {self.high}]")
        return coerced

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "default": self.default,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "choices": list(self.choices) if self.choices else None,
        }


@dataclass(frozen=True)
class Output:
    kind: OutputKind
    lo: float | None = None
    hi: float | None = None
    values: tuple[int, ...] | None = None  # for STATE outputs
    description: str = ""
    # Params this output depends on (None = all). Unused params are stripped from feature
    # specs, so logically identical strategies hash identically and the GA never mutates
    # knobs that cannot change the result.
    uses: tuple[str, ...] | None = None


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    fn: Callable[..., Outputs]
    category: Category
    params: dict[str, Param]
    outputs: dict[str, Output]
    description: str
    needs_calendar: bool = False

    def resolve(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        return {k: p.validate(k, params.get(k, p.default)) for k, p in self.params.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "params": {k: p.to_dict() for k, p in self.params.items()},
            "outputs": {
                k: {
                    "kind": o.kind.value,
                    "lo": o.lo,
                    "hi": o.hi,
                    "values": list(o.values) if o.values else None,
                    "description": o.description,
                }
                for k, o in self.outputs.items()
            },
        }


REGISTRY: dict[str, IndicatorSpec] = {}


def indicator(
    name: str,
    *,
    category: Category,
    params: dict[str, Param] | None = None,
    outputs: dict[str, Output],
    description: str,
    needs_calendar: bool = False,
) -> Callable[[Callable[..., Outputs]], Callable[..., Outputs]]:
    def register(fn: Callable[..., Outputs]) -> Callable[..., Outputs]:
        if name in REGISTRY:
            raise ValueError(f"indicator {name!r} registered twice")
        REGISTRY[name] = IndicatorSpec(
            name, fn, category, params or {}, outputs, description, needs_calendar
        )
        return fn

    return register


def get_spec(name: str) -> IndicatorSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown indicator {name!r}; known: {sorted(REGISTRY)}") from exc


@dataclass
class Inputs:
    """Price arrays (mid prices) plus calendar context for one bar series.

    ``cached`` memoizes intermediates shared between indicators (ATR, pivots, RSI...),
    so a feature set with ten structure-based indicators computes pivots once.
    """

    ts: pd.DatetimeIndex  # bar open time, UTC
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    spread: FloatArray
    timeframe: Timeframe
    calendar: MarketCalendar | None = None
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _cache: dict[Hashable, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_bars(
        cls,
        bars: pd.DataFrame,
        timeframe: Timeframe,
        calendar: MarketCalendar | None = None,
        sessions: dict[str, Any] | None = None,
    ) -> Inputs:
        def mid(f: str) -> FloatArray:
            return ((bars[f"bid_{f}"] + bars[f"ask_{f}"]) / 2.0).to_numpy(np.float64)

        return cls(
            ts=pd.DatetimeIndex(bars["ts"]),
            open=mid("open"),
            high=mid("high"),
            low=mid("low"),
            close=mid("close"),
            volume=bars["volume"].to_numpy(np.float64),
            spread=(bars["ask_close"] - bars["bid_close"]).to_numpy(np.float64),
            timeframe=timeframe,
            calendar=calendar,
            sessions={k: dict(v) for k, v in (sessions or {}).items()},
        )

    def __len__(self) -> int:
        return len(self.close)

    def cached(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def slice(self, stop: int) -> Inputs:
        """First ``stop`` bars. Used by the lookahead tests."""
        return Inputs(
            self.ts[:stop],
            self.open[:stop],
            self.high[:stop],
            self.low[:stop],
            self.close[:stop],
            self.volume[:stop],
            self.spread[:stop],
            self.timeframe,
            self.calendar,
            self.sessions,
        )


def compute(inputs: Inputs, name: str, params: dict[str, Any] | None = None) -> Outputs:
    spec = get_spec(name)
    resolved = spec.resolve(params)
    if spec.needs_calendar and inputs.calendar is None:
        raise ValueError(f"{name} needs a market calendar")
    key = ("indicator", name, tuple(sorted(resolved.items())))
    result: Outputs = inputs.cached(key, lambda: spec.fn(inputs, **resolved))
    missing = set(spec.outputs) - set(result)
    if missing:  # pragma: no cover - guards indicator implementations
        raise RuntimeError(f"{name} did not produce outputs {sorted(missing)}")
    return {k: result[k] for k in spec.outputs}


@dataclass(frozen=True)
class FeatureRef:
    """A specific output of an indicator with specific params, e.g. ``rsi(length=14).value``."""

    indicator: str
    output: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def of(cls, indicator: str, output: str, **params: Any) -> FeatureRef:
        resolved = get_spec(indicator).resolve(params)
        if output not in get_spec(indicator).outputs:
            raise ValueError(f"{indicator} has no output {output!r}")
        return cls(indicator, output, tuple(sorted(resolved.items())))

    @property
    def column(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.indicator}({args}).{self.output}"

    @property
    def kind(self) -> OutputKind:
        return get_spec(self.indicator).outputs[self.output].kind


def relevant_params(indicator: str, output: str) -> tuple[str, ...]:
    spec = get_spec(indicator)
    uses = spec.outputs[output].uses
    return tuple(spec.params) if uses is None else uses


def spec_hash(items: Any) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:16]
