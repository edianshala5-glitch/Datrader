"""Classic indicators on mid prices. Formulas follow TradingView Pine (``ta.*``) so values
can be checked against charts. Wilder RMA and EMA are SMA-seeded, Bollinger uses
population stdev."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.indicators.base import (
    Category,
    FloatArray,
    Inputs,
    Output,
    OutputKind,
    Outputs,
    Param,
    indicator,
)
from engine.indicators.kernels import seeded_ema

PRICE = Output(OutputKind.PRICE)
PCT = Output(OutputKind.OSCILLATOR, 0, 100)


# --------------------------------------------------------------------------- primitives
def ema(x: FloatArray, n: int) -> FloatArray:
    return seeded_ema(np.asarray(x, np.float64), n, 2.0 / (n + 1))


def rma(x: FloatArray, n: int) -> FloatArray:
    return seeded_ema(np.asarray(x, np.float64), n, 1.0 / n)


def sma(x: FloatArray, n: int) -> FloatArray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def rolling_max(x: FloatArray, n: int) -> FloatArray:
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(x: FloatArray, n: int) -> FloatArray:
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def true_range(inp: Inputs) -> FloatArray:
    def build() -> FloatArray:
        prev = np.concatenate([[np.nan], inp.close[:-1]])
        tr = np.fmax(inp.high - inp.low, np.fmax(np.abs(inp.high - prev), np.abs(inp.low - prev)))
        out: FloatArray = tr  # first bar: high - low (fmax ignores the NaN)
        return out

    return inp.cached(("tr",), build)  # type: ignore[no-any-return]


def atr_values(inp: Inputs, length: int = 14) -> FloatArray:
    return inp.cached(("atr", length), lambda: rma(true_range(inp), length))  # type: ignore[no-any-return]


def rsi_values(src: FloatArray, length: int) -> FloatArray:
    change = np.diff(src, prepend=np.nan)
    up = rma(np.where(np.isnan(change), np.nan, np.maximum(change, 0.0)), length)
    down = rma(np.where(np.isnan(change), np.nan, np.maximum(-change, 0.0)), length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + up / down)
    rsi = np.where(down == 0, 100.0, rsi)
    rsi = np.where((up == 0) & (down > 0), 0.0, rsi)
    return np.where(np.isnan(up) | np.isnan(down), np.nan, rsi)


def macd_values(src: FloatArray, fast: int, slow: int, signal: int) -> tuple[FloatArray, ...]:
    line = ema(src, fast) - ema(src, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


# --------------------------------------------------------------------------- trend
@indicator(
    "ema",
    category=Category.TREND,
    params={"length": Param(21, 2, 400, 1)},
    outputs={"value": PRICE},
    description="Exponential moving average of close (SMA-seeded, as TradingView).",
)
def _ema(inp: Inputs, length: int) -> Outputs:
    return {"value": ema(inp.close, length)}


@indicator(
    "sma",
    category=Category.TREND,
    params={"length": Param(50, 2, 400, 1)},
    outputs={"value": PRICE},
    description="Simple moving average of close.",
)
def _sma(inp: Inputs, length: int) -> Outputs:
    return {"value": sma(inp.close, length)}


@indicator(
    "macd",
    category=Category.MOMENTUM,
    params={
        "fast": Param(12, 2, 50, 1),
        "slow": Param(26, 5, 120, 1),
        "signal": Param(9, 2, 50, 1),
    },
    outputs={
        "line": Output(OutputKind.UNBOUNDED),
        "signal": Output(OutputKind.UNBOUNDED),
        "hist": Output(OutputKind.UNBOUNDED),
    },
    description="MACD line, signal and histogram.",
)
def _macd(inp: Inputs, fast: int, slow: int, signal: int) -> Outputs:
    if fast >= slow:
        raise ValueError("macd: fast must be < slow")
    line, sig, hist = inp.cached(
        ("macd", fast, slow, signal), lambda: macd_values(inp.close, fast, slow, signal)
    )
    return {"line": line, "signal": sig, "hist": hist}


@indicator(
    "adx",
    category=Category.TREND,
    params={"length": Param(14, 2, 100, 1)},
    outputs={
        "adx": PCT,
        "plus_di": PCT,
        "minus_di": PCT,
    },
    description="Wilder's ADX with +DI / -DI (trend strength and direction).",
)
def _adx(inp: Inputs, length: int) -> Outputs:
    up = np.diff(inp.high, prepend=np.nan)
    down = -np.diff(inp.low, prepend=np.nan)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan
    tr = true_range(inp).copy()
    tr[0] = np.nan
    trur = rma(tr, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100.0 * rma(plus_dm, length) / trur
        minus = 100.0 * rma(minus_dm, length) / trur
        dx = np.abs(plus - minus) / np.where(plus + minus == 0, 1.0, plus + minus)
    return {"adx": 100.0 * rma(dx, length), "plus_di": plus, "minus_di": minus}


@indicator(
    "donchian",
    category=Category.TREND,
    params={"length": Param(20, 2, 400, 1)},
    outputs={"upper": PRICE, "lower": PRICE, "mid": PRICE},
    description="Donchian channel: highest high / lowest low over `length` bars (incl. current).",
)
def _donchian(inp: Inputs, length: int) -> Outputs:
    up, lo = rolling_max(inp.high, length), rolling_min(inp.low, length)
    return {"upper": up, "lower": lo, "mid": (up + lo) / 2}


# --------------------------------------------------------------------------- momentum
@indicator(
    "rsi",
    category=Category.MOMENTUM,
    params={"length": Param(14, 2, 100, 1)},
    outputs={"value": PCT},
    description="Wilder RSI of close.",
)
def _rsi(inp: Inputs, length: int) -> Outputs:
    return {"value": inp.cached(("rsi", length), lambda: rsi_values(inp.close, length))}


@indicator(
    "stoch",
    category=Category.MOMENTUM,
    params={
        "length": Param(14, 2, 100, 1),
        "smooth_k": Param(3, 1, 10, 1),
        "smooth_d": Param(3, 1, 10, 1),
    },
    outputs={"k": PCT, "d": PCT},
    description="Stochastic oscillator %K / %D.",
)
def _stoch(inp: Inputs, length: int, smooth_k: int, smooth_d: int) -> Outputs:
    hh, ll = rolling_max(inp.high, length), rolling_min(inp.low, length)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(rng > 0, 100.0 * (inp.close - ll) / rng, 50.0)
    raw = np.where(np.isnan(hh), np.nan, raw)
    k = sma(raw, smooth_k)
    return {"k": k, "d": sma(k, smooth_d)}


@indicator(
    "roc",
    category=Category.MOMENTUM,
    params={"length": Param(10, 1, 200, 1)},
    outputs={"value": Output(OutputKind.UNBOUNDED, description="percent change")},
    description="Rate of change in percent over `length` bars.",
)
def _roc(inp: Inputs, length: int) -> Outputs:
    prev = np.concatenate([np.full(length, np.nan), inp.close[:-length]])
    return {"value": 100.0 * (inp.close / prev - 1.0)}


# --------------------------------------------------------------------------- volatility
@indicator(
    "atr",
    category=Category.VOLATILITY,
    params={"length": Param(14, 2, 100, 1)},
    outputs={
        "value": Output(OutputKind.UNBOUNDED, description="price units"),
        "pct": Output(OutputKind.RATIO, description="ATR as % of close"),
    },
    description="Wilder Average True Range.",
)
def _atr(inp: Inputs, length: int) -> Outputs:
    a = atr_values(inp, length)
    return {"value": a, "pct": 100.0 * a / inp.close}


@indicator(
    "bbands",
    category=Category.VOLATILITY,
    params={"length": Param(20, 5, 200, 1), "mult": Param(2.0, 0.5, 4.0, 0.1)},
    outputs={
        "upper": PRICE,
        "mid": PRICE,
        "lower": PRICE,
        "pct_b": Output(OutputKind.OSCILLATOR, -0.5, 1.5, description="0=lower, 1=upper"),
        "width": Output(OutputKind.RATIO, description="(upper-lower)/mid"),
    },
    description="Bollinger Bands (population stdev, as TradingView).",
)
def _bbands(inp: Inputs, length: int, mult: float) -> Outputs:
    s = pd.Series(inp.close)
    mid = s.rolling(length, min_periods=length).mean().to_numpy()
    dev = mult * s.rolling(length, min_periods=length).std(ddof=0).to_numpy()
    up, lo = mid + dev, mid - dev
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(up > lo, (inp.close - lo) / (up - lo), 0.5)
    pct_b = np.where(np.isnan(mid), np.nan, pct_b)
    return {"upper": up, "mid": mid, "lower": lo, "pct_b": pct_b, "width": (up - lo) / mid}


# --------------------------------------------------------------------------- volume
@indicator(
    "vwap",
    category=Category.VOLUME,
    outputs={
        "value": PRICE,
        "dist_atr": Output(OutputKind.RATIO, description="(close - vwap) / ATR(14)"),
    },
    description="Session VWAP anchored at the trading-day roll (17:00 NY for gold). "
    "Falls back to equal weights when the source has no volume.",
    needs_calendar=True,
)
def _vwap(inp: Inputs) -> Outputs:
    assert inp.calendar is not None
    day = inp.calendar.trading_day(inp.ts).asi8
    tp = (inp.high + inp.low + inp.close) / 3.0
    vol = np.where(inp.volume > 0, inp.volume, 0.0)
    df = pd.DataFrame({"day": day, "pv": tp * vol, "v": vol, "tp": tp, "one": 1.0})
    g = df.groupby("day", sort=False)
    cum_v = g["v"].cumsum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vw = np.where(cum_v > 0, g["pv"].cumsum().to_numpy() / cum_v, np.nan)
    equal = g["tp"].cumsum().to_numpy() / g["one"].cumsum().to_numpy()
    vw = np.where(np.isnan(vw), equal, vw)
    return {"value": vw, "dist_atr": (inp.close - vw) / atr_values(inp, 14)}
