"""Performance metrics from a trade list and a bar-level equity curve.

Return-based ratios (Sharpe, Sortino) use **daily** returns, where a day is the gold trading
day (17:00 NY roll), annualized with sqrt(252). Bar-level returns would overstate Sharpe for
intraday strategies through autocorrelation and the many flat bars.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _finite(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def daily_returns(equity: pd.Series, trading_day: pd.DatetimeIndex, initial: float) -> pd.Series:
    """Close-of-day equity → daily simple returns (first day measured against ``initial``)."""
    eod = equity.groupby(np.asarray(trading_day)).last()
    prev = eod.shift(1)
    prev.iloc[0] = initial
    return (eod / prev - 1.0).astype(float)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    trading_day: pd.DatetimeIndex,
    initial_equity: float,
    position: np.ndarray | None = None,
) -> dict[str, Any]:
    n = len(trades)
    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    total_return = final / initial_equity - 1.0
    span_days = (
        (equity.index[-1] - equity.index[0]).total_seconds() / 86400 if len(equity) > 1 else 0.0
    )
    years = span_days / 365.25
    cagr = (final / initial_equity) ** (1 / years) - 1 if years > 0 and final > 0 else math.nan

    dr = (
        daily_returns(equity, trading_day, initial_equity)
        if len(equity)
        else pd.Series(dtype=float)
    )
    sharpe = sortino = math.nan
    if len(dr) >= 2:
        sd = dr.std(ddof=1)
        if sd > 0:
            sharpe = dr.mean() / sd * math.sqrt(TRADING_DAYS)
        downside = math.sqrt(float(np.mean(np.minimum(dr.to_numpy(), 0.0) ** 2)))
        if downside > 0:
            sortino = dr.mean() / downside * math.sqrt(TRADING_DAYS)
    mdd = max_drawdown(equity.to_numpy())
    calmar = cagr / mdd if mdd > 0 and math.isfinite(cagr) else math.nan

    m: dict[str, Any] = {
        "n_trades": n,
        "total_return": _finite(total_return),
        "cagr": _finite(cagr),
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "max_drawdown": _finite(mdd),
        "calmar": _finite(calmar),
        "trading_days": len(dr),
        "exposure": _finite(float(np.mean(position != 0)))
        if position is not None and len(position)
        else None,
    }
    if n == 0:
        m.update(
            profit_factor=None,
            win_rate=None,
            expectancy_r=None,
            avg_bars_held=None,
            avg_win_r=None,
            avg_loss_r=None,
            max_consecutive_losses=0,
            sqn=None,
            gross_profit=0.0,
            gross_loss=0.0,
            total_commission=0.0,
            long_trades=0,
            short_trades=0,
            trades_per_week=0.0,
            exit_reasons={},
        )
        return m

    pnl = trades["pnl"].to_numpy()
    r = trades["r"].to_numpy()
    wins, losses = pnl > 0, pnl <= 0
    gp, gl = float(pnl[wins].sum()), float(-pnl[losses].sum())
    sqn = math.sqrt(n) * r.mean() / r.std(ddof=1) if n > 1 and r.std(ddof=1) > 0 else math.nan
    m.update(
        profit_factor=_finite(gp / gl) if gl > 0 else (None if gp == 0 else 99.0),
        win_rate=float(wins.mean()),
        expectancy_r=_finite(float(r.mean())),
        avg_bars_held=float((trades["exit_idx"] - trades["entry_idx"] + 1).mean()),
        avg_win_r=_finite(float(r[wins].mean())) if wins.any() else None,
        avg_loss_r=_finite(float(r[losses].mean())) if losses.any() else None,
        max_consecutive_losses=max_consecutive(losses),
        sqn=_finite(sqn),
        gross_profit=gp,
        gross_loss=gl,
        total_commission=float(trades["commission"].sum()),
        long_trades=int((trades["side"] == "long").sum()),
        short_trades=int((trades["side"] == "short").sum()),
        trades_per_week=n / (span_days / 7) if span_days > 0 else None,
        exit_reasons=trades["exit_reason"].value_counts().to_dict(),
    )
    return m
