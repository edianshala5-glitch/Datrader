from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.db.types import UTCDateTime
from app.models import genome_hash, strategy_codename
from app.models.enums import PRIORITY_TIMEFRAMES, Timeframe
from app.schemas.instrument import CostModel, InstrumentCreate, SessionWindow
from app.services.seed import XAUUSD

sqlite = __import__("sqlalchemy.dialects.sqlite", fromlist=["dialect"]).dialect()


class TestUTCDateTime:
    def test_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            UTCDateTime().process_bind_param(datetime(2024, 1, 2, 9, 30), sqlite)

    def test_normalizes_to_utc(self) -> None:
        ny_winter = timezone(timedelta(hours=-5))
        value = datetime(2024, 1, 2, 9, 30, tzinfo=ny_winter)
        bound = UTCDateTime().process_bind_param(value, sqlite)
        assert bound == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
        assert bound is not None and bound.tzinfo == UTC

    def test_tags_naive_results_as_utc(self) -> None:
        out = UTCDateTime().process_result_value(datetime(2024, 1, 2, 14, 30), sqlite)
        assert out == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)

    def test_parses_sqlite_string_results(self) -> None:
        out = UTCDateTime().process_result_value("2024-01-02 14:30:00", sqlite)
        assert out == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def test_timeframe_seconds_and_priority() -> None:
    assert Timeframe.M5.seconds == 300
    assert Timeframe.M30.seconds == 1800
    assert PRIORITY_TIMEFRAMES == (Timeframe.M5, Timeframe.M15, Timeframe.M30)


def test_genome_hash_is_key_order_invariant() -> None:
    a = {"entry": {"rsi": {"period": 14, "below": 30}}, "exit": {"rr": 2.0}}
    b = {"exit": {"rr": 2.0}, "entry": {"rsi": {"below": 30, "period": 14}}}
    assert genome_hash(a) == genome_hash(b)
    assert genome_hash(a) != genome_hash({**a, "exit": {"rr": 2.5}})
    assert strategy_codename(genome_hash(a)).startswith("AU-")


class TestInstrumentSchemas:
    def test_xauusd_definition(self) -> None:
        assert XAUUSD.tick_size == Decimal("0.01")
        assert XAUUSD.contract_size == Decimal("100")
        assert XAUUSD.session_timezone == "America/New_York"
        assert XAUUSD.trading_hours.day_roll.hour == 17
        assert {"london", "new_york", "london_kz", "ny_am_kz"} <= XAUUSD.sessions.keys()

    def test_rejects_unknown_timezone(self) -> None:
        with pytest.raises(ValidationError, match="unknown IANA timezone"):
            SessionWindow(label="x", tz="Mars/Olympus", start="08:00", end="09:00")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            InstrumentCreate(**{**XAUUSD.model_dump(), "session_timezone": "EST5EDT-ish"})

    def test_rejects_negative_costs(self) -> None:
        with pytest.raises(ValidationError):
            CostModel(spread=-0.1, commission_per_lot_round_turn=7, slippage_ticks=1)

    def test_symbol_pattern(self) -> None:
        with pytest.raises(ValidationError):
            InstrumentCreate(**{**XAUUSD.model_dump(), "symbol": "xau usd"})


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWARM_DATABASE_URL", "postgresql+asyncpg://u:p@db/x")
    monkeypatch.setenv("SWARM_CORS_ORIGINS", '["http://a", "http://b"]')
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert not s.is_sqlite
    assert s.cors_origins == ["http://a", "http://b"]
    assert s.redis_url is None
