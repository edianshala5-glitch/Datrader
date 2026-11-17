"""Time-series data splits: research period with walk-forward folds, plus a locked holdout.

    |<-------------------- research -------------------->|<-- holdout -->|
    | train 1 |e| test 1 |
    |     train 2     |e| test 2 |                        (anchored)
    |           train 3          |e| test 3 |
                                                         ^ evaluated once per strategy

``e`` is an embargo gap between train and test. Trades or indicator state from the end of the
train window can't bleed into the test window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
            for k, v in asdict(self).items()
        }


@dataclass(frozen=True)
class DataSplits:
    research_start: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    folds: tuple[Fold, ...]
    embargo: pd.Timedelta

    def to_dict(self) -> dict[str, Any]:
        """Stored on ``EvolutionRun.data_splits``."""
        return {
            "research_start": self.research_start.isoformat(),
            "holdout_start": self.holdout_start.isoformat(),
            "holdout_end": self.holdout_end.isoformat(),
            "embargo_hours": self.embargo.total_seconds() / 3600,
            "folds": [f.to_dict() for f in self.folds],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataSplits:
        ts = pd.Timestamp
        return cls(
            research_start=ts(d["research_start"]),
            holdout_start=ts(d["holdout_start"]),
            holdout_end=ts(d["holdout_end"]),
            embargo=pd.Timedelta(hours=d["embargo_hours"]),
            folds=tuple(
                Fold(
                    f["index"],
                    ts(f["train_start"]),
                    ts(f["train_end"]),
                    ts(f["test_start"]),
                    ts(f["test_end"]),
                )
                for f in d["folds"]
            ),
        )


def _ts(v: datetime | str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def make_splits(
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    *,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    anchored: bool = True,
    min_train_fraction: float = 0.4,
    embargo: pd.Timedelta = pd.Timedelta(days=1),
) -> DataSplits:
    """Split [start, end) into a research period and a final holdout of ``holdout_fraction``.

    The research period gets ``n_folds`` walk-forward folds. The first train window covers
    ``min_train_fraction`` of the research period, and the remainder is cut into equal test
    windows. ``anchored=True`` grows the train window from the start. Otherwise it rolls
    with a fixed length.
    """
    s, e = _ts(start), _ts(end)
    if not 0 <= holdout_fraction < 0.9:
        raise ValueError("holdout_fraction must be in [0, 0.9)")
    if n_folds < 1:
        raise ValueError("need at least one fold")
    total = e - s
    holdout_start = e - total * holdout_fraction
    research = holdout_start - s
    first_train = research * min_train_fraction
    test_len = (research - first_train) / n_folds
    if test_len <= embargo * 2:
        raise ValueError("research period too short for this many folds and embargo")
    folds = []
    for i in range(n_folds):
        test_start = s + first_train + test_len * i
        test_end = test_start + test_len if i < n_folds - 1 else holdout_start
        train_end = test_start - embargo
        # rolling: constant train length (first train window minus the embargo)
        train_start = s if anchored else max(s, train_end - (first_train - embargo))
        folds.append(Fold(i, train_start, train_end, test_start, test_end))
    return DataSplits(s, holdout_start, e, tuple(folds), embargo)


def block_edges(start: pd.Timestamp, end: pd.Timestamp, n_blocks: int) -> list[pd.Timestamp]:
    step = (end - start) / n_blocks
    return [start + step * i for i in range(n_blocks + 1)]
