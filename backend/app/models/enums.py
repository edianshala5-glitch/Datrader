"""Domain enums shared by models, schemas and the quant engine."""

from __future__ import annotations

from enum import StrEnum


class AssetClass(StrEnum):
    METAL = "metal"
    FX = "fx"
    INDEX = "index"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class Timeframe(StrEnum):
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def seconds(self) -> int:
        return _TIMEFRAME_SECONDS[self]


_TIMEFRAME_SECONDS = {
    Timeframe.TICK: 0,
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3600,
    Timeframe.H4: 14400,
    Timeframe.D1: 86400,
}

# The research focus. The swarm runs these first.
PRIORITY_TIMEFRAMES = (Timeframe.M5, Timeframe.M15, Timeframe.M30)


class DataSource(StrEnum):
    DUKASCOPY = "dukascopy"  # free tick history (bid/ask), XAUUSD back to ~2003
    HISTDATA = "histdata"  # free M1 CSV archives
    CSV = "csv"  # user-supplied files (e.g. MT5 exports)
    OANDA = "oanda"  # free practice-account API (live stream)
    SYNTHETIC = "synthetic"  # generated data for tests


class JobKind(StrEnum):
    INGEST = "ingest"
    EVOLUTION = "evolution"
    VALIDATION = "validation"
    META_ANALYSIS = "meta_analysis"
    FEATURES = "features"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StrategyOrigin(StrEnum):
    SEED = "seed"  # hand-written template
    RANDOM = "random"  # random initial population
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    MIGRANT = "migrant"  # arrived from another island
    MANUAL = "manual"
    RL = "rl"


class StrategyStatus(StrEnum):
    CANDIDATE = "candidate"  # evaluated in-sample only
    VALIDATED = "validated"  # passed walk-forward + statistical gates
    REJECTED = "rejected"  # failed a gate. Kept for lineage and multiple-testing counts.
    INCUBATING = "incubating"  # paper trading on live data before becoming champion
    CHAMPION = "champion"
    RETIRED = "retired"


class Segment(StrEnum):
    IN_SAMPLE = "in_sample"
    WALK_FORWARD = "walk_forward"  # one row per out-of-sample fold
    VALIDATION = "validation"
    HOLDOUT = "holdout"  # locked final period, evaluated once per strategy
    INCUBATION = "incubation"  # forward paper trading
    LIVE = "live"


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class ExitReason(StrEnum):
    STOP = "stop"
    TARGET = "target"
    TRAILING_STOP = "trailing_stop"
    SIGNAL = "signal"
    TIME = "time"  # max bars in trade
    SESSION_CLOSE = "session_close"
    END_OF_DATA = "end_of_data"


class RuleStatus(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ACTIVE = "active"
    RETIRED = "retired"


class ChampionStatus(StrEnum):
    INCUBATING = "incubating"
    ACTIVE = "active"
    DEMOTED = "demoted"


class SignalKind(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"
    CANCEL = "cancel"


class SignalStatus(StrEnum):
    PENDING = "pending"  # fills at next bar open
    ACTIVE = "active"  # position assumed open
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    CLOSED = "closed"  # closed by exit signal / time / session
    EXPIRED = "expired"  # entry never filled
    CANCELLED = "cancelled"
