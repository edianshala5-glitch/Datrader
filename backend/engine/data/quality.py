"""Data-quality report for M1 bars, stored on ``Dataset.quality`` and shown in the UI.

Checks:
* coverage: share of expected open-market minutes that have a bar
* gaps: runs of missing open-market minutes (US-holiday / year-end gaps counted separately)
* bars while the calendar says closed: calendar mismatch or bad data
* duplicates / non-monotonic timestamps
* crossed quotes (ask < bid) and inconsistent OHLC
* spread distribution (median / p95 / p99 / max)
* price spikes: robust z-score of mid log-returns (median/MAD), immune to the outliers it hunts
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.data.calendar import MarketCalendar
from engine.data.holidays import is_us_holiday
from engine.data.schema import mid

# Gold trades thinly or not at all around Christmas / New Year; gaps there are expected.
_HOLIDAY_WINDOWS = ((12, 24, 12, 26), (12, 31, 1, 2))


def _is_holiday_gap(ts_local: pd.Timestamp) -> bool:
    """Gap starting on a US market holiday (NY date) or in the year-end season."""
    if is_us_holiday(ts_local.date()):
        return True
    md = (ts_local.month, ts_local.day)
    for m1, d1, m2, d2 in _HOLIDAY_WINDOWS:
        inside = (m1, d1) <= md <= (m2, d2) if m1 <= m2 else md >= (m1, d1) or md <= (m2, d2)
        if inside:
            return True
    return False


def _r(x: float | np.floating[Any] | None, nd: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def m1_quality_report(
    bars: pd.DataFrame,
    calendar: MarketCalendar,
    *,
    gap_min_minutes: int = 5,
    spike_z: float = 15.0,
    max_examples: int = 20,
) -> dict[str, Any]:
    if bars.empty:
        return {"rows": 0, "score": 0.0}

    ts = bars["ts"]
    start, end = ts.iloc[0], ts.iloc[-1]
    report: dict[str, Any] = {
        "rows": len(bars),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duplicates": int(ts.duplicated().sum()),
        "non_monotonic": int((ts.diff().dt.total_seconds() < 0).sum()),
    }

    # --- quote sanity
    crossed = (bars["ask_close"] < bars["bid_close"]) | (bars["ask_low"] < bars["bid_low"])
    bad_ohlc = (bars["bid_high"] < bars[["bid_open", "bid_close"]].max(axis=1)) | (
        bars["bid_low"] > bars[["bid_open", "bid_close"]].min(axis=1)
    )
    report["crossed_quotes"] = int(crossed.sum())
    report["invalid_ohlc"] = int(bad_ohlc.sum())

    spread = bars["ask_close"] - bars["bid_close"]
    report["spread"] = {
        "median": _r(spread.median()),
        "p95": _r(spread.quantile(0.95)),
        "p99": _r(spread.quantile(0.99)),
        "max": _r(spread.max()),
    }

    # --- coverage and gaps against the calendar
    expected = calendar.open_minutes(start, end + pd.Timedelta(minutes=1))
    present = pd.DatetimeIndex(ts.drop_duplicates())
    open_mask = calendar.is_open(present)
    report["bars_while_closed"] = int((~open_mask).sum())
    missing = expected.difference(present)
    report["expected_minutes"] = len(expected)
    report["missing_minutes"] = len(missing)
    report["coverage"] = _r(1 - len(missing) / len(expected)) if len(expected) else 1.0

    gaps: list[dict[str, Any]] = []
    holiday_gaps = 0
    if len(missing):
        # Consecutive missing minutes (in the expected grid) form one gap.
        pos = expected.get_indexer(missing)
        breaks = np.flatnonzero(np.diff(pos) != 1) + 1
        for run in np.split(np.arange(len(missing)), breaks):
            if len(run) < gap_min_minutes:
                continue
            g_start = missing[run[0]]
            if _is_holiday_gap(g_start.tz_convert(calendar.tz)):
                holiday_gaps += 1
                continue
            gaps.append(
                {
                    "start": g_start.isoformat(),
                    "end": (missing[run[-1]] + pd.Timedelta(minutes=1)).isoformat(),
                    "minutes": len(run),
                }
            )
    gaps.sort(key=lambda g: g["minutes"], reverse=True)
    report["gaps"] = {
        "count": len(gaps),
        "holiday": holiday_gaps,
        "largest": gaps[:max_examples],
    }

    # --- spikes vs. genuine large moves
    # z = |r| / local volatility (rolling median |r| over the previous day of bars), so
    # busy London/NY sessions are judged against busy-session volatility. A *spike* is a
    # large move that mostly reverses on the next bar (the signature of a bad print). A
    # large move that holds (NFP, FOMC) is real and reported separately.
    m = mid(bars).to_numpy()
    rets = np.diff(np.log(m)) if len(m) > 1 else np.zeros(0)
    spikes: list[dict[str, Any]] = []
    report["spike_count"] = 0
    report["large_moves"] = 0
    if len(rets) > 10:
        abs_r = pd.Series(np.abs(rets))
        rolling = abs_r.shift(1).rolling(1440, min_periods=60).median() * 1.4826
        local = rolling.fillna(abs_r.median() * 1.4826).replace(0, np.nan).to_numpy()
        z = np.abs(rets) / local
        big = np.nan_to_num(z) > spike_z
        nxt = np.append(rets[1:], 0.0)
        reverts = (np.sign(nxt) == -np.sign(rets)) & (np.abs(nxt) >= 0.7 * np.abs(rets))
        spike_idx = np.flatnonzero(big & reverts)
        report["spike_count"] = len(spike_idx)
        report["large_moves"] = int((big & ~reverts).sum())
        order = spike_idx[np.argsort(-z[spike_idx])][:max_examples]
        spikes = [
            {"ts": ts.iloc[i + 1].isoformat(), "ret": _r(rets[i], 6), "z": _r(z[i], 1)}
            for i in order
        ]
    report["spikes"] = spikes

    # --- headline score 0..1 (coverage dominates, hard errors penalize)
    hard_errors = report["duplicates"] + report["non_monotonic"] + report["crossed_quotes"]
    score = float(report["coverage"] or 0.0)
    score -= min(0.3, hard_errors / max(len(bars), 1) * 100)
    score -= min(0.1, report["spike_count"] / max(len(bars), 1) * 1000)
    report["score"] = _r(max(0.0, score), 3)
    return report
