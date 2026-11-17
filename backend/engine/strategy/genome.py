"""Strategy genome: the JSON DSL every strategy is written in.

Design goals:
* **Typed against the indicator registry.** Conditions are checked against each output's
  kind (event, flag, state, oscillator, price...), so the GA can never produce
  meaningless rules such as "EMA > 37" or "RSI > 140".
* **Serializable and hashable.** The canonical JSON is stored on ``strategies.genome``
  and de-duplicated by hash.
* **Readable.** :mod:`engine.strategy.explain` renders it as plain English for the UI and for
  live-signal rationale.

Example (abridged)::

    {"entry_long": {"type": "group", "op": "all", "conditions": [
        {"type": "event", "within": 5,
         "feature": {"indicator": "structure", "output": "choch_bull"}},
        {"type": "compare", "op": "<",
         "left": {"kind": "feature", "feature": {"indicator": "rsi", "output": "value"}},
         "right": {"kind": "const", "value": 45}},
        {"type": "flag", "feature": {"indicator": "sessions", "output": "ny_am_kz"}}]},
     "exit": {"stop": {"type": "atr", "mult": 1.5}, "target": {"type": "rr", "ratio": 2.0},
              "flat_at": "daily"}}
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import Timeframe
from engine.indicators import FeatureRef, OutputKind, get_spec
from engine.indicators.base import relevant_params

MAX_DEPTH = 3
MAX_CONDITIONS = 12

NUMERIC_KINDS = {
    OutputKind.PRICE,
    OutputKind.OSCILLATOR,
    OutputKind.UNBOUNDED,
    OutputKind.RATIO,
    OutputKind.STATE,
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureSpec(_Strict):
    indicator: str
    output: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeframe: Timeframe | None = None  # higher timeframe; None = the strategy's own

    @model_validator(mode="before")
    @classmethod
    def _strip_unused(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and data.get("params")
            and "indicator" in data
            and "output" in data
        ):
            try:
                keep = set(relevant_params(data["indicator"], data["output"]))
            except KeyError:
                return data  # unknown indicator/output: let the after-validator report it
            data = {**data, "params": {k: v for k, v in data["params"].items() if k in keep}}
        return data

    @model_validator(mode="after")
    def _check(self) -> FeatureSpec:
        spec = get_spec(self.indicator)
        if self.output not in spec.outputs:
            raise ValueError(f"{self.indicator} has no output {self.output!r}")
        spec.resolve(self.params)  # raises on unknown / out-of-range params
        if self.timeframe is Timeframe.TICK:
            raise ValueError("tick is not a feature timeframe")
        return self

    def __hash__(self) -> int:
        return hash(
            (self.indicator, self.output, tuple(sorted(self.params.items())), self.timeframe)
        )

    @property
    def ref(self) -> FeatureRef:
        return FeatureRef.of(self.indicator, self.output, **self.params)

    @property
    def kind(self) -> OutputKind:
        return get_spec(self.indicator).outputs[self.output].kind

    def column(self) -> str:
        col = self.ref.column
        return f"{self.timeframe.value}:{col}" if self.timeframe else col


# --------------------------------------------------------------------------- operands
class FeatureOperand(_Strict):
    kind: Literal["feature"] = "feature"
    feature: FeatureSpec
    shift: int = Field(0, ge=0, le=50)  # value `shift` bars ago


class PriceOperand(_Strict):
    kind: Literal["price"] = "price"
    field: Literal["open", "high", "low", "close"] = "close"
    shift: int = Field(0, ge=0, le=50)


class ConstOperand(_Strict):
    kind: Literal["const"] = "const"
    value: float


Operand = Annotated[FeatureOperand | PriceOperand | ConstOperand, Field(discriminator="kind")]


def _unit(op: FeatureOperand | PriceOperand | ConstOperand) -> str:
    if isinstance(op, ConstOperand):
        return "const"
    if isinstance(op, PriceOperand) or op.feature.kind is OutputKind.PRICE:
        return "price"
    return f"value:{op.feature.kind.value}"


def _check_comparable(left: Any, right: Any) -> None:
    lu, ru = _unit(left), _unit(right)
    if lu == "const" and ru == "const":
        raise ValueError("comparing two constants is meaningless")
    if "price" in (lu, ru) and lu != ru:
        raise ValueError("price-level values can only be compared with other price levels")
    if lu.startswith("value") and ru.startswith("value") and lu != ru:
        raise ValueError(f"cannot compare {lu.split(':')[1]} with {ru.split(':')[1]}")
    for op, other in ((left, right), (right, left)):
        if isinstance(op, FeatureOperand) and isinstance(other, ConstOperand):
            decl = get_spec(op.feature.indicator).outputs[op.feature.output]
            if op.feature.kind not in NUMERIC_KINDS:
                raise ValueError(f"{op.feature.column()} is not numeric")
            if (decl.lo is not None and other.value < decl.lo) or (
                decl.hi is not None and other.value > decl.hi
            ):
                raise ValueError(
                    f"threshold {other.value} outside {op.feature.column()} range "
                    f"[{decl.lo}, {decl.hi}]"
                )
        if isinstance(op, FeatureOperand) and op.feature.kind not in NUMERIC_KINDS:
            raise ValueError(f"{op.feature.column()} ({op.feature.kind}) cannot be compared")


# --------------------------------------------------------------------------- conditions
class Compare(_Strict):
    type: Literal["compare"] = "compare"
    left: Operand
    op: Literal[">", "<", ">=", "<="]
    right: Operand

    @model_validator(mode="after")
    def _check(self) -> Compare:
        _check_comparable(self.left, self.right)
        return self


class Cross(_Strict):
    type: Literal["cross"] = "cross"
    left: Operand
    right: Operand
    direction: Literal["above", "below"]

    @model_validator(mode="after")
    def _check(self) -> Cross:
        _check_comparable(self.left, self.right)
        return self


class EventCond(_Strict):
    """``feature`` fired on this bar (within=0) or within the last ``within`` bars."""

    type: Literal["event"] = "event"
    feature: FeatureSpec
    within: int = Field(0, ge=0, le=200)

    @model_validator(mode="after")
    def _check(self) -> EventCond:
        if self.feature.kind not in (OutputKind.EVENT, OutputKind.FLAG):
            raise ValueError(f"{self.feature.column()} is not an event")
        return self


class FlagCond(_Strict):
    type: Literal["flag"] = "flag"
    feature: FeatureSpec
    value: bool = True

    @model_validator(mode="after")
    def _check(self) -> FlagCond:
        if self.feature.kind not in (OutputKind.FLAG, OutputKind.EVENT):
            raise ValueError(f"{self.feature.column()} is not a flag")
        return self


class StateCond(_Strict):
    type: Literal["state"] = "state"
    feature: FeatureSpec
    values: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> StateCond:
        decl = get_spec(self.feature.indicator).outputs[self.feature.output]
        if decl.kind is not OutputKind.STATE:
            raise ValueError(f"{self.feature.column()} is not a state")
        if decl.values and not set(self.values) <= set(decl.values):
            raise ValueError(f"values {self.values} not in {list(decl.values)}")
        return self


class RangeCond(_Strict):
    """lo <= feature <= hi (e.g. NY minute-of-day between 09:30 and 11:00)."""

    type: Literal["range"] = "range"
    feature: FeatureSpec
    lo: float
    hi: float

    @model_validator(mode="after")
    def _check(self) -> RangeCond:
        if self.lo > self.hi:
            raise ValueError("range lo > hi")
        if self.feature.kind not in NUMERIC_KINDS - {OutputKind.PRICE}:
            raise ValueError(f"{self.feature.column()} cannot be range-tested")
        return self


class Group(_Strict):
    type: Literal["group"] = "group"
    op: Literal["all", "any"] = "all"
    negate: bool = False
    conditions: list[Condition] = Field(min_length=1, max_length=8)


Condition = Annotated[
    Compare | Cross | EventCond | FlagCond | StateCond | RangeCond | Group,
    Field(discriminator="type"),
]
Group.model_rebuild()


def walk(cond: Any, depth: int = 1) -> Iterator[tuple[Any, int]]:
    yield cond, depth
    if isinstance(cond, Group):
        for c in cond.conditions:
            yield from walk(c, depth + 1)


# --------------------------------------------------------------------------- exits & risk
class AtrStop(_Strict):
    type: Literal["atr"] = "atr"
    mult: float = Field(1.5, ge=0.2, le=10)
    length: int = Field(14, ge=2, le=100)


class SwingStop(_Strict):
    """Beyond the last confirmed swing low (longs) / high (shorts), plus a buffer."""

    type: Literal["swing"] = "swing"
    length: int = Field(5, ge=2, le=30)
    buffer_atr: float = Field(0.1, ge=0, le=2)
    max_atr: float = Field(4.0, ge=0.5, le=15)  # skip trades whose stop would be wider


class PercentStop(_Strict):
    type: Literal["percent"] = "percent"
    pct: float = Field(0.3, ge=0.01, le=5)


StopRule = Annotated[AtrStop | SwingStop | PercentStop, Field(discriminator="type")]


class RRTarget(_Strict):
    type: Literal["rr"] = "rr"
    ratio: float = Field(2.0, ge=0.3, le=10)


class AtrTarget(_Strict):
    type: Literal["atr"] = "atr"
    mult: float = Field(3.0, ge=0.2, le=20)
    length: int = Field(14, ge=2, le=100)


TargetRule = Annotated[RRTarget | AtrTarget, Field(discriminator="type")]


class AtrTrail(_Strict):
    type: Literal["atr"] = "atr"
    mult: float = Field(2.0, ge=0.2, le=10)
    length: int = Field(14, ge=2, le=100)
    activate_r: float = Field(1.0, ge=0, le=5)  # start trailing once +activate_r R in profit


class ExitRules(_Strict):
    stop: StopRule = Field(default_factory=AtrStop)
    target: TargetRule | None = Field(default_factory=RRTarget)
    trailing: AtrTrail | None = None
    max_bars: int | None = Field(None, ge=1, le=5000)
    signal_long: Group | None = None  # close longs when true
    signal_short: Group | None = None
    flat_at: Literal["daily", "weekend", "none"] = "daily"  # day trading: flat before 17:00 NY


class RiskRules(_Strict):
    risk_per_trade: float = Field(0.005, ge=0.0005, le=0.05)  # fraction of equity
    max_lots: float = Field(50.0, gt=0, le=1000)


class Genome(_Strict):
    version: Literal[1] = 1
    name: str | None = None
    entry_long: Group | None = None
    entry_short: Group | None = None
    filter: Group | None = None  # applies to both directions
    exit: ExitRules = Field(default_factory=ExitRules)
    risk: RiskRules = Field(default_factory=RiskRules)

    @model_validator(mode="after")
    def _check(self) -> Genome:
        if self.entry_long is None and self.entry_short is None:
            raise ValueError("a genome needs entry_long and/or entry_short")
        total = 0
        for root in self.condition_roots():
            for node, depth in walk(root):
                if depth > MAX_DEPTH:
                    raise ValueError(f"condition tree deeper than {MAX_DEPTH}")
                if not isinstance(node, Group):
                    total += 1
        if total > MAX_CONDITIONS:
            raise ValueError(f"{total} conditions exceed the limit of {MAX_CONDITIONS}")
        return self

    def condition_roots(self) -> list[Group]:
        roots = [
            self.entry_long,
            self.entry_short,
            self.filter,
            self.exit.signal_long,
            self.exit.signal_short,
        ]
        return [r for r in roots if r is not None]

    def features(self) -> set[FeatureSpec]:
        """Every feature the genome reads (excluding the ATR/swing data used by exits)."""
        out: set[FeatureSpec] = set()
        for root in self.condition_roots():
            for node, _ in walk(root):
                for attr in ("feature", "left", "right"):
                    v = getattr(node, attr, None)
                    if isinstance(v, FeatureSpec):
                        out.add(v)
                    elif isinstance(v, FeatureOperand):
                        out.add(v.feature)
        return out

    def canonical(self) -> dict[str, Any]:
        """JSON form stored in the DB and hashed for de-duplication.

        ``None`` values are kept on purpose. ``target: null`` means "no target", and dropping
        it would silently turn into the default 2R target on reload.
        """
        return self.model_dump(mode="json")
