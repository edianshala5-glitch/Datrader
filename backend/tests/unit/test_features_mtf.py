from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.data.calendar import MarketCalendar
from engine.data.quality import m1_quality_report
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.indicators import FeatureRef, Inputs
from engine.indicators import features as feat
from engine.indicators.mtf import align_htf

CAL = MarketCalendar.xauusd()
SESSIONS = XAUUSD.model_dump(mode="json")["sessions"]


@pytest.fixture(scope="module")
def m1() -> pd.DataFrame:
    return generate_m1(datetime(2024, 3, 3, tzinfo=UTC), datetime(2024, 3, 16, tzinfo=UTC), CAL)


def test_align_htf_never_uses_incomplete_bars() -> None:
    ltf = pd.date_range("2024-03-05 10:00", "2024-03-05 11:45", freq="15min", tz="UTC")
    htf = pd.date_range("2024-03-05 08:00", "2024-03-05 12:00", freq="1h", tz="UTC")
    vals = pd.DataFrame({"hour_open": htf.hour})
    out = align_htf(ltf, Timeframe.M15, htf, Timeframe.H1, vals)
    seen = dict(zip(ltf.strftime("%H:%M"), out["hour_open"], strict=True))
    # 10:00-10:45 bars decide by 11:00 at the latest; the 10:00 hour completes at 11:00.
    assert seen["10:00"] == 9 and seen["10:30"] == 9
    assert seen["10:45"] == 10  # closes 11:00 exactly when the 10:00 hour completes
    assert seen["11:15"] == 10 and seen["11:45"] == 11


def test_align_htf_rejects_lower_tf() -> None:
    with pytest.raises(ValueError):
        align_htf(
            pd.DatetimeIndex([]), Timeframe.H1, pd.DatetimeIndex([]), Timeframe.M15, pd.DataFrame()
        )


def test_add_htf_features_is_leak_free(m1: pd.DataFrame) -> None:
    m15 = resample_bars(m1, Timeframe.M15, CAL)
    h1 = resample_bars(m1, Timeframe.H1, CAL)
    reqs = [feat.FeatureRequest.of("ema", length=10)]
    f15 = feat.build_features(Inputs.from_bars(m15, Timeframe.M15, CAL), reqs)
    f1h = feat.build_features(Inputs.from_bars(h1, Timeframe.H1, CAL), reqs)
    joined = feat.add_htf_features(f15, Timeframe.M15, f1h, Timeframe.H1)
    col = "1h:ema(length=10).value"
    assert col in joined.columns and len(joined) == len(f15)
    # Check every row: the HTF value used must come from an hour complete by the decision.
    h1_avail = f1h["ts"] + pd.Timedelta(hours=1)
    decide = joined["ts"] + pd.Timedelta(minutes=15)
    pos = np.searchsorted(h1_avail.to_numpy(), decide.to_numpy(), side="right") - 1
    expected = np.where(
        pos >= 0, f1h["ema(length=10).value"].to_numpy()[np.maximum(pos, 0)], np.nan
    )
    np.testing.assert_allclose(joined[col].to_numpy(), expected, equal_nan=True)


def test_feature_refs_and_columns() -> None:
    ref = FeatureRef.of("rsi", "value")
    assert ref.column == "rsi(length=14).value"
    assert FeatureRef.of("structure", "choch_bull", length=8).column == (
        "structure(break_on=close,eq_tol=0.1,length=8).choch_bull"
    )
    with pytest.raises(ValueError):
        FeatureRef.of("rsi", "nope")


def test_default_feature_matrix_and_bars_since(m1: pd.DataFrame) -> None:
    m5 = resample_bars(m1, Timeframe.M5, CAL)
    frame = feat.build_features(
        Inputs.from_bars(m5, Timeframe.M5, CAL, SESSIONS), feat.default_requests()
    )
    assert frame.shape[0] == len(m5) and frame.shape[1] > 80
    ev = "structure(break_on=close,eq_tol=0.1,length=5).choch_bull"
    out = feat.with_bars_since(frame, [ev])
    since = out[f"{ev}.bars_since"].to_numpy()
    assert (since[frame[ev].to_numpy()] == 0).all()


def test_feature_cache_hit(
    m1: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m15 = resample_bars(m1, Timeframe.M15, CAL)
    cache = feat.FeatureCache(tmp_path)
    reqs = [feat.FeatureRequest.of("rsi"), feat.FeatureRequest.of("structure")]
    first = cache.get_or_build("XAUUSD", "synthetic", Timeframe.M15, m15, reqs, CAL)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("should have been served from cache")

    monkeypatch.setattr(feat, "build_features", boom)
    again = cache.get_or_build("XAUUSD", "synthetic", Timeframe.M15, m15, reqs, CAL)
    pd.testing.assert_frame_equal(first, again)
    # Different data → different key → recompute
    assert cache.path("XAUUSD", "synthetic", Timeframe.M15, reqs, m15.iloc[:-1]) != cache.path(
        "XAUUSD", "synthetic", Timeframe.M15, reqs, m15
    )


def test_quality_report_treats_mlk_early_close_as_holiday() -> None:
    week = generate_m1(datetime(2024, 1, 14, tzinfo=UTC), datetime(2024, 1, 20, tzinfo=UTC), CAL)
    ny = week["ts"].dt.tz_convert("America/New_York")
    early_close = (
        (ny.dt.date == pd.Timestamp("2024-01-15").date())
        & (ny.dt.hour >= 14)
        & (ny.dt.hour < 17)
        & ~((ny.dt.hour == 14) & (ny.dt.minute < 30))
    )
    r = m1_quality_report(week[~early_close].reset_index(drop=True), CAL)
    assert r["gaps"]["count"] == 0 and r["gaps"]["holiday"] == 1
