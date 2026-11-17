"""Resampling, Parquet store, quality report, CSV import and synthetic generator."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.enums import DataSource, Timeframe
from engine.data.calendar import MarketCalendar
from engine.data.quality import m1_quality_report
from engine.data.resample import resample_bars, ticks_to_m1
from engine.data.schema import BAR_COLUMNS
from engine.data.sources import dukascopy as d
from engine.data.sources.csv import parse_bars_csv
from engine.data.sources.synthetic import generate_m1
from engine.data.store import ParquetStore

CAL = MarketCalendar.xauusd()
FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "dukascopy" / "XAUUSD_2024-01-02_14h_ticks.bi5"
)
HOUR = datetime(2024, 1, 2, 14, tzinfo=UTC)
SYN = DataSource.SYNTHETIC


@pytest.fixture(scope="module")
def ticks() -> pd.DataFrame:
    return d.decode_ticks(FIXTURE.read_bytes(), HOUR, 1000)


@pytest.fixture(scope="module")
def week() -> pd.DataFrame:
    return generate_m1(datetime(2024, 3, 3, tzinfo=UTC), datetime(2024, 3, 10, tzinfo=UTC), CAL)


# --------------------------------------------------------------------------- resample
def test_ticks_to_m1_matches_manual_aggregation(ticks: pd.DataFrame) -> None:
    m1 = ticks_to_m1(ticks)
    assert list(m1.columns) == BAR_COLUMNS and len(m1) == 60
    first_min = ticks[ticks["ts"] < pd.Timestamp(HOUR) + pd.Timedelta(minutes=1)]
    row = m1.iloc[0]
    assert row["bid_open"] == first_min["bid"].iloc[0]
    assert row["bid_high"] == first_min["bid"].max()
    assert row["ask_low"] == first_min["ask"].min()
    assert row["bid_close"] == first_min["bid"].iloc[-1]
    assert row["tick_count"] == len(first_min)
    assert m1["tick_count"].sum() == len(ticks)


def test_resample_is_consistent_across_paths(ticks: pd.DataFrame) -> None:
    """M1→15m must equal aggregating the raw ticks straight into 15m buckets."""
    m15 = resample_bars(ticks_to_m1(ticks), Timeframe.M15, CAL)
    assert len(m15) == 4
    t = ticks.assign(bucket=ticks["ts"].dt.floor("15min"))
    direct = t.groupby("bucket").agg(hi=("bid", "max"), lo=("ask", "min"), close=("bid", "last"))
    assert np.allclose(m15["bid_high"], direct["hi"])
    assert np.allclose(m15["ask_low"], direct["lo"])
    assert np.allclose(m15["bid_close"], direct["close"])
    assert m15["tick_count"].sum() == len(ticks)


def test_daily_bars_follow_ny_close(week: pd.DataFrame) -> None:
    d1 = resample_bars(week, Timeframe.D1, CAL)
    # Week of 2024-03-03 (EST): five sessions opening 22:00 UTC Sun..Thu.
    assert d1["ts"].dt.strftime("%a %H:%M").tolist() == [
        "Sun 22:00",
        "Mon 22:00",
        "Tue 22:00",
        "Wed 22:00",
        "Thu 22:00",
    ]
    assert d1["volume"].sum() == pytest.approx(week["volume"].sum())


def test_resample_rejects_m1_target(week: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        resample_bars(week, Timeframe.M1, CAL)


# --------------------------------------------------------------------------- store
def test_store_roundtrip_merge_and_tail(tmp_path: Path, week: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path)
    half = len(week) // 2
    store.write_bars("XAUUSD", SYN, Timeframe.M1, week.iloc[: half + 100])
    store.write_bars("XAUUSD", SYN, Timeframe.M1, week.iloc[half:])  # overlapping write
    back = store.read_bars("XAUUSD", Timeframe.M1, source=SYN)
    pd.testing.assert_frame_equal(back, week.reset_index(drop=True))

    window = store.read_bars(
        "XAUUSD", Timeframe.M1, source=SYN, start="2024-03-05 14:00", end="2024-03-05 15:00"
    )
    assert len(window) == 60 and window["ts"].iloc[0] == pd.Timestamp("2024-03-05 14:00", tz="UTC")

    tail = store.read_bars("XAUUSD", Timeframe.M1, source=SYN, limit=3, tail=True)
    assert tail["ts"].tolist() == week["ts"].iloc[-3:].tolist()

    cov = store.bar_coverage("XAUUSD", Timeframe.M1, SYN)
    assert (cov.rows, cov.start, cov.end) == (len(week), week["ts"].iloc[0], week["ts"].iloc[-1])
    assert store.read_bars("XAUUSD", Timeframe.H1, source=SYN).empty  # nothing written


def test_store_partitions_by_year(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    bars = generate_m1(datetime(2023, 12, 28, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC), CAL)
    store.write_bars("XAUUSD", SYN, Timeframe.M1, bars)
    assert store.bar_years("XAUUSD", Timeframe.M1, SYN) == [2023, 2024]
    jan = store.read_bars("XAUUSD", Timeframe.M1, source=SYN, start="2024-01-01")
    assert jan["ts"].min().year == 2024


def test_store_ticks_replace_whole_hours(tmp_path: Path, ticks: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path)
    store.write_ticks("XAUUSD", DataSource.DUKASCOPY, ticks)
    store.write_ticks("XAUUSD", DataSource.DUKASCOPY, ticks)  # re-download same hour
    back = store.read_ticks("XAUUSD")
    assert len(back) == len(ticks)  # replaced, not duplicated
    assert back["ts"].iloc[0] == ticks["ts"].iloc[0]  # ms precision survives storage
    assert store.tick_coverage("XAUUSD").rows == len(ticks)


# --------------------------------------------------------------------------- quality
def test_quality_clean_week(week: pd.DataFrame) -> None:
    r = m1_quality_report(week, CAL)
    assert r["coverage"] == 1.0 and r["gaps"]["count"] == 0
    assert r["duplicates"] == r["crossed_quotes"] == r["bars_while_closed"] == 0
    assert r["score"] > 0.95


def test_quality_detects_problems(week: pd.DataFrame) -> None:
    bad = week.copy()
    gap = (bad["ts"] >= "2024-03-05 14:00") & (bad["ts"] < "2024-03-05 14:30")
    bad = bad[~gap].reset_index(drop=True)
    bad.loc[100, ["ask_close", "ask_low"]] = bad.loc[100, "bid_close"] - 1  # crossed quote
    # Bad print: +3% then straight back.
    for col in (
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ):
        bad.loc[2000, col] = bad.loc[2000, col] * 1.03
    closed_bar = bad.iloc[[0]].assign(ts=pd.Timestamp("2024-03-09 12:00", tz="UTC"))  # Saturday
    bad = pd.concat([bad, bad.iloc[[50]], closed_bar]).sort_values("ts").reset_index(drop=True)

    r = m1_quality_report(bad, CAL)
    assert r["gaps"]["count"] == 1 and r["gaps"]["largest"][0]["minutes"] == 30
    assert r["crossed_quotes"] >= 1
    assert r["duplicates"] == 1
    assert r["bars_while_closed"] == 1
    assert r["spike_count"] >= 1
    assert r["score"] < 1.0


# --------------------------------------------------------------------------- csv
MT5 = (
    "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
    "2024.01.03\t16:30:00\t2050.10\t2051.00\t2049.90\t2050.50\t120\t0\t25\n"
    "2024.07.03\t16:30:00\t2330.10\t2331.00\t2329.90\t2330.50\t140\t0\t30\n"
)


def test_mt5_csv_with_ny_close_server_time() -> None:
    bars = parse_bars_csv(MT5, tz="NY+7", point=0.01)
    # Server 16:30 = NY 09:30 → 14:30 UTC in winter, 13:30 UTC in summer.
    assert bars["ts"].dt.strftime("%m-%d %H:%M").tolist() == ["01-03 14:30", "07-03 13:30"]
    assert bars["ask_close"].round(2).tolist() == [2050.75, 2330.80]
    assert bars["volume"].tolist() == [120.0, 140.0]


def test_generic_csv() -> None:
    text = "time,open,high,low,close,volume\n2024-01-03T14:30:00Z,1,2,0.5,1.5,10\n"
    bars = parse_bars_csv(text)
    assert bars["ts"].iloc[0] == pd.Timestamp("2024-01-03 14:30", tz="UTC")
    assert (bars["ask_close"] == bars["bid_close"]).all()  # no spread column
    with pytest.raises(ValueError, match="unrecognized"):
        parse_bars_csv("a,b\n1,2\n")


# --------------------------------------------------------------------------- synthetic
def test_synthetic_is_deterministic_and_calendar_aware(week: pd.DataFrame) -> None:
    again = generate_m1(datetime(2024, 3, 3, tzinfo=UTC), datetime(2024, 3, 10, tzinfo=UTC), CAL)
    pd.testing.assert_frame_equal(week, again)
    assert CAL.is_open(week["ts"]).all()
    assert (week["ask_low"] > week["bid_low"]).all()
    assert (week["bid_high"] >= week[["bid_open", "bid_close"]].max(axis=1)).all()
