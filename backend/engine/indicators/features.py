"""Feature matrices: many indicator outputs over one bar series, cached on disk.

A strategy genome (Step 4) references features as :class:`FeatureRef` columns. The swarm
evaluates thousands of genomes over the *same* few hundred features. Computing each feature
once per (symbol, timeframe, data version) and caching it as Parquet is what makes that
affordable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import Timeframe
from engine.data.calendar import MarketCalendar
from engine.indicators.base import REGISTRY, FeatureRef, Inputs, compute, get_spec, spec_hash
from engine.indicators.kernels import bars_since
from engine.indicators.mtf import align_htf

logger = logging.getLogger(__name__)

ENGINE_VERSION = "3.0"  # bump to invalidate every cached feature matrix


@dataclass(frozen=True)
class FeatureRequest:
    indicator: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def of(cls, indicator: str, **params: Any) -> FeatureRequest:
        return cls(indicator, tuple(sorted(get_spec(indicator).resolve(params).items())))

    def refs(self) -> list[FeatureRef]:
        return [
            FeatureRef(self.indicator, out, self.params) for out in get_spec(self.indicator).outputs
        ]


def default_requests() -> list[FeatureRequest]:
    """Every registered indicator at default params (the Meta-Analyzer's context set)."""
    return [FeatureRequest.of(name) for name in sorted(REGISTRY)]


def build_features(inputs: Inputs, requests: Iterable[FeatureRequest]) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {}
    for req in requests:
        outs = compute(inputs, req.indicator, dict(req.params))
        for ref in req.refs():
            columns[ref.column] = outs[ref.output]
    frame = pd.DataFrame(columns)
    frame.insert(0, "ts", inputs.ts)
    return frame


def with_bars_since(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add ``<col>.bars_since`` for event columns (e.g. bars since the last bullish CHoCH)."""
    out = features.copy()
    for c in columns:
        out[f"{c}.bars_since"] = bars_since(features[c].to_numpy(dtype=bool))
    return out


def add_htf_features(
    ltf: pd.DataFrame,
    ltf_tf: Timeframe,
    htf: pd.DataFrame,
    htf_tf: Timeframe,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join HTF feature columns onto LTF rows (leak-free), prefixed ``<htf>:``."""
    cols = [c for c in (columns or htf.columns) if c != "ts"]
    aligned = align_htf(ltf["ts"], ltf_tf, htf["ts"], htf_tf, htf[cols])
    aligned.columns = [f"{htf_tf.value}:{c}" for c in cols]
    return pd.concat([ltf.reset_index(drop=True), aligned], axis=1)


class FeatureCache:
    """Parquet cache keyed by (symbol, source, timeframe, requests, data fingerprint)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def fingerprint(bars: pd.DataFrame) -> dict[str, Any]:
        if bars.empty:
            return {"rows": 0}
        return {
            "rows": len(bars),
            "first": bars["ts"].iloc[0].isoformat(),
            "last": bars["ts"].iloc[-1].isoformat(),
            "close_sum": round(float(bars["bid_close"].sum()), 4),
        }

    def path(
        self,
        symbol: str,
        source: str,
        timeframe: Timeframe,
        requests: Sequence[FeatureRequest],
        bars: pd.DataFrame,
    ) -> Path:
        key = spec_hash(
            {
                "v": ENGINE_VERSION,
                "req": [(r.indicator, r.params) for r in requests],
                "data": self.fingerprint(bars),
            }
        )
        return self.root / symbol / source / timeframe.value / f"{key}.parquet"

    def get_or_build(
        self,
        symbol: str,
        source: str,
        timeframe: Timeframe,
        bars: pd.DataFrame,
        requests: Sequence[FeatureRequest],
        calendar: MarketCalendar | None,
        sessions: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        path = self.path(symbol, source, timeframe, requests, bars)
        if path.exists():
            return pd.read_parquet(path)
        inputs = Inputs.from_bars(bars, timeframe, calendar, sessions)
        frame = build_features(inputs, requests)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        frame.to_parquet(tmp, compression="zstd", index=False)
        tmp.replace(path)
        logger.info("feature matrix cached", extra={"path": str(path), "cols": frame.shape[1]})
        return frame
