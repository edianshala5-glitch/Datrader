"""Multi-timeframe alignment without leakage.

A higher-timeframe bar opening at ``T`` is only *complete* at ``T + htf``. A lower-timeframe
bar opening at ``t`` makes decisions at its close ``t + ltf``. So the LTF bar may use the
latest HTF bar whose completion time is ``<= t + ltf``.

Example (15m strategy, 1h structure): the 10:45-11:00 bar can see the 10:00-11:00 hourly
bar (complete at 11:00). The 10:30-10:45 bar still sees only the 09:00-10:00 hour.
"""

from __future__ import annotations

import pandas as pd

from app.models.enums import Timeframe


def align_htf(
    ltf_ts: pd.DatetimeIndex | pd.Series,
    ltf_tf: Timeframe,
    htf_ts: pd.DatetimeIndex | pd.Series,
    htf_tf: Timeframe,
    htf_values: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``htf_values`` re-indexed onto the LTF bars (one row per LTF bar)."""
    if htf_tf.seconds <= ltf_tf.seconds:
        raise ValueError(f"{htf_tf} is not a higher timeframe than {ltf_tf}")
    left = pd.DataFrame(
        {"_decide": pd.DatetimeIndex(ltf_ts) + pd.Timedelta(seconds=ltf_tf.seconds)}
    )
    left["_order"] = range(len(left))
    right = htf_values.reset_index(drop=True).copy()
    right["_avail"] = pd.DatetimeIndex(htf_ts) + pd.Timedelta(seconds=htf_tf.seconds)
    merged = pd.merge_asof(
        left.sort_values("_decide"),
        right.sort_values("_avail"),
        left_on="_decide",
        right_on="_avail",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_order")
    return merged.drop(columns=["_decide", "_order", "_avail"]).reset_index(drop=True)
