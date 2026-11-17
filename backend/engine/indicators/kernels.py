"""numba kernels for the sequential parts of the indicator engine.

Everything here is causal: loops run forward in time and write output ``t`` using only
inputs ``<= t``. Swing pivots are the one place that needs future bars. Their outputs are
therefore written at the *confirmation* bar ``i + right``, never at the pivot bar ``i``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

NAN = np.nan


# --------------------------------------------------------------------------- smoothing
@njit(cache=True)
def seeded_ema(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """EMA-family smoother seeded with the SMA of the first ``n`` valid values. This matches
    TradingView ``ta.ema``/``ta.rma`` and TA-Lib. Leading NaNs are skipped. Later NaNs
    carry the previous value forward."""
    out = np.full(x.shape[0], NAN)
    start = 0
    while start < x.shape[0] and np.isnan(x[start]):
        start += 1
    if x.shape[0] - start < n:
        return out
    s = 0.0
    for i in range(start, start + n):
        s += x[i]
    prev = s / n
    out[start + n - 1] = prev
    for i in range(start + n, x.shape[0]):
        if not np.isnan(x[i]):
            prev = alpha * x[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


# --------------------------------------------------------------------------- swings
@njit(cache=True)
def pivots(high: np.ndarray, low: np.ndarray, left: int, right: int):  # type: ignore[no-untyped-def]
    """Fractal pivots. A pivot high at ``i`` has strictly lower highs on the ``left`` bars
    before it and lower-or-equal highs on the ``right`` bars after (so equal highs yield
    one pivot, the first). Returns boolean arrays indexed by *pivot* bar. Callers must
    shift to confirmation time (``i + right``) before exposing anything."""
    n = high.shape[0]
    ph = np.zeros(n, dtype=np.bool_)
    pl = np.zeros(n, dtype=np.bool_)
    for i in range(left, n - right):
        h = high[i]
        ok = True
        for j in range(i - left, i):
            if high[j] >= h:
                ok = False
                break
        if ok:
            for j in range(i + 1, i + right + 1):
                if high[j] > h:
                    ok = False
                    break
        ph[i] = ok
        lo = low[i]
        ok = True
        for j in range(i - left, i):
            if low[j] <= lo:
                ok = False
                break
        if ok:
            for j in range(i + 1, i + right + 1):
                if low[j] < lo:
                    ok = False
                    break
        pl[i] = ok
    return ph, pl


@njit(cache=True)
def confirmed_swings(high: np.ndarray, low: np.ndarray, left: int, right: int):  # type: ignore[no-untyped-def]
    """Swing info as known at each bar (confirmation-time view).

    Returns per bar ``t``:
      sh_event, sl_event   True if a swing high/low was confirmed at t
      sh_price, sl_price   its price (NaN otherwise)
      sh_pivot, sl_pivot   its pivot bar index (-1 otherwise)
    """
    n = high.shape[0]
    ph, pl = pivots(high, low, left, right)
    sh_event = np.zeros(n, dtype=np.bool_)
    sl_event = np.zeros(n, dtype=np.bool_)
    sh_price = np.full(n, NAN)
    sl_price = np.full(n, NAN)
    sh_pivot = np.full(n, -1, dtype=np.int64)
    sl_pivot = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        c = i + right
        if c >= n:
            break
        if ph[i]:
            sh_event[c] = True
            sh_price[c] = high[i]
            sh_pivot[c] = i
        if pl[i]:
            sl_event[c] = True
            sl_price[c] = low[i]
            sl_pivot[c] = i
    return sh_event, sl_event, sh_price, sl_price, sh_pivot, sl_pivot


# --------------------------------------------------------------------------- structure
@njit(cache=True)
def market_structure(  # type: ignore[no-untyped-def]
    high, low, close, atr, sh_event, sl_event, sh_price, sl_price, use_close, eq_tol_atr
):
    """SMC market structure from confirmed swings.

    * BOS: break of the latest swing in the direction of the current trend.
    * CHoCH: break against the current trend, which flips it. The first break from an
      undefined trend is a BOS.
    * Sweep: a wick through an unbroken swing level whose close comes back inside (a
      liquidity grab). It fires once per level, and the level stays live: a later close
      beyond it is still a BOS/CHoCH (e.g. a sweep before NFP, then the real break).
    * EQH/EQL: a new swing within ``eq_tol_atr`` x ATR of the previous one (resting
      liquidity).
    """
    n = close.shape[0]
    trend = np.zeros(n, dtype=np.int8)
    bos_bull = np.zeros(n, dtype=np.bool_)
    bos_bear = np.zeros(n, dtype=np.bool_)
    choch_bull = np.zeros(n, dtype=np.bool_)
    choch_bear = np.zeros(n, dtype=np.bool_)
    sweep_high = np.zeros(n, dtype=np.bool_)
    sweep_low = np.zeros(n, dtype=np.bool_)
    eqh = np.zeros(n, dtype=np.bool_)
    eql = np.zeros(n, dtype=np.bool_)
    level_high = np.full(n, NAN)
    level_low = np.full(n, NAN)

    cur = 0
    lh = NAN
    ll = NAN
    lh_live = False
    ll_live = False
    lh_swept = False
    ll_swept = False
    prev_sh = NAN
    prev_sl = NAN
    for t in range(n):
        up_px = close[t] if use_close else high[t]
        dn_px = close[t] if use_close else low[t]
        if lh_live and up_px > lh:
            if cur == -1:
                choch_bull[t] = True
            else:
                bos_bull[t] = True
            cur = 1
            lh_live = False
        elif lh_live and not lh_swept and high[t] > lh:
            sweep_high[t] = True
            lh_swept = True
        if ll_live and dn_px < ll:
            if cur == 1:
                choch_bear[t] = True
            else:
                bos_bear[t] = True
            cur = -1
            ll_live = False
        elif ll_live and not ll_swept and low[t] < ll:
            sweep_low[t] = True
            ll_swept = True

        # Swings confirmed at t become levels for later bars. (Bar t cannot break its own
        # newly confirmed swing: confirmation implies high[t] <= swing high.)
        if sh_event[t]:
            tol = eq_tol_atr * atr[t] if not np.isnan(atr[t]) else 0.0
            if not np.isnan(prev_sh) and abs(sh_price[t] - prev_sh) <= tol:
                eqh[t] = True
            prev_sh = sh_price[t]
            lh = sh_price[t]
            lh_live = True
            lh_swept = False
        if sl_event[t]:
            tol = eq_tol_atr * atr[t] if not np.isnan(atr[t]) else 0.0
            if not np.isnan(prev_sl) and abs(sl_price[t] - prev_sl) <= tol:
                eql[t] = True
            prev_sl = sl_price[t]
            ll = sl_price[t]
            ll_live = True
            ll_swept = False

        trend[t] = cur
        level_high[t] = lh if lh_live else NAN
        level_low[t] = ll if ll_live else NAN
    return (
        trend,
        bos_bull,
        bos_bear,
        choch_bull,
        choch_bear,
        sweep_high,
        sweep_low,
        eqh,
        eql,
        level_high,
        level_low,
    )


# --------------------------------------------------------------------------- liquidity
@njit(cache=True)
def fair_value_gaps(high, low, close, atr, min_size_atr, max_active):  # type: ignore[no-untyped-def]
    """Three-candle imbalances, known at the close of the third candle.

    Bullish FVG at t: low[t] > high[t-2], zone [high[t-2], low[t]]. A zone dies when price
    fills it completely (low <= bottom for bullish) or closes through it.
    ``touch_*`` fires when a later bar trades into a live zone without closing through it.
    ``end_idx`` (bar where each zone died) is for chart overlays only. It is future
    information and must never be used as a feature.
    """
    n = close.shape[0]
    new_bull = np.zeros(n, dtype=np.bool_)
    new_bear = np.zeros(n, dtype=np.bool_)
    touch_bull = np.zeros(n, dtype=np.bool_)
    touch_bear = np.zeros(n, dtype=np.bool_)
    bull_top = np.full(n, NAN)
    bull_bot = np.full(n, NAN)
    bear_top = np.full(n, NAN)
    bear_bot = np.full(n, NAN)
    n_bull = np.zeros(n, dtype=np.int32)
    n_bear = np.zeros(n, dtype=np.int32)
    end_idx = np.full(n, -1, dtype=np.int64)

    zb_top = np.empty(max_active)
    zb_bot = np.empty(max_active)
    zb_t = np.empty(max_active, dtype=np.int64)
    kb = 0
    zs_top = np.empty(max_active)
    zs_bot = np.empty(max_active)
    zs_t = np.empty(max_active, dtype=np.int64)
    ks = 0

    for t in range(n):
        # 1) update live zones with bar t
        j = 0
        while j < kb:
            dead = low[t] <= zb_bot[j] or close[t] < zb_bot[j]
            if low[t] <= zb_top[j] and not dead:
                touch_bull[t] = True
            if dead:
                end_idx[zb_t[j]] = t
                kb -= 1
                zb_top[j] = zb_top[kb]
                zb_bot[j] = zb_bot[kb]
                zb_t[j] = zb_t[kb]
            else:
                j += 1
        j = 0
        while j < ks:
            dead = high[t] >= zs_top[j] or close[t] > zs_top[j]
            if high[t] >= zs_bot[j] and not dead:
                touch_bear[t] = True
            if dead:
                end_idx[zs_t[j]] = t
                ks -= 1
                zs_top[j] = zs_top[ks]
                zs_bot[j] = zs_bot[ks]
                zs_t[j] = zs_t[ks]
            else:
                j += 1

        # 2) new zones completed by bar t
        if t >= 2:
            min_gap = min_size_atr * atr[t] if not np.isnan(atr[t]) else 0.0
            gap = low[t] - high[t - 2]
            if gap > 0 and gap >= min_gap:
                new_bull[t] = True
                if kb == max_active:  # drop the oldest
                    oldest = 0
                    for q in range(1, kb):
                        if zb_t[q] < zb_t[oldest]:
                            oldest = q
                    kb -= 1
                    zb_top[oldest] = zb_top[kb]
                    zb_bot[oldest] = zb_bot[kb]
                    zb_t[oldest] = zb_t[kb]
                zb_top[kb] = low[t]
                zb_bot[kb] = high[t - 2]
                zb_t[kb] = t
                kb += 1
            gap = low[t - 2] - high[t]
            if gap > 0 and gap >= min_gap:
                new_bear[t] = True
                if ks == max_active:
                    oldest = 0
                    for q in range(1, ks):
                        if zs_t[q] < zs_t[oldest]:
                            oldest = q
                    ks -= 1
                    zs_top[oldest] = zs_top[ks]
                    zs_bot[oldest] = zs_bot[ks]
                    zs_t[oldest] = zs_t[ks]
                zs_top[ks] = low[t - 2]
                zs_bot[ks] = high[t]
                zs_t[ks] = t
                ks += 1

        # 3) nearest live zones: bullish = highest top at/below close, bearish = lowest
        # bottom at/above close
        best = -1
        for q in range(kb):
            if zb_top[q] <= close[t] and (best < 0 or zb_top[q] > zb_top[best]):
                best = q
        if best >= 0:
            bull_top[t] = zb_top[best]
            bull_bot[t] = zb_bot[best]
        best = -1
        for q in range(ks):
            if zs_bot[q] >= close[t] and (best < 0 or zs_bot[q] < zs_bot[best]):
                best = q
        if best >= 0:
            bear_top[t] = zs_top[best]
            bear_bot[t] = zs_bot[best]
        n_bull[t] = kb
        n_bear[t] = ks
    return (
        new_bull,
        new_bear,
        touch_bull,
        touch_bear,
        bull_top,
        bull_bot,
        bear_top,
        bear_bot,
        n_bull,
        n_bear,
        end_idx,
    )


@njit(cache=True)
def order_blocks(open_, high, low, close, bull_break, bear_break, lookback):  # type: ignore[no-untyped-def]
    """Order blocks anchored on structure breaks.

    On a bullish BOS/CHoCH at t, the bullish OB is the most recent down-close candle in
    the ``lookback`` bars before t (the last selling before the displacement). It stays live
    until a close below its low. ``touch_bull`` fires when a later bar trades into it.
    Only the most recent OB per side is tracked.
    """
    n = close.shape[0]
    new_bull = np.zeros(n, dtype=np.bool_)
    new_bear = np.zeros(n, dtype=np.bool_)
    touch_bull = np.zeros(n, dtype=np.bool_)
    touch_bear = np.zeros(n, dtype=np.bool_)
    bull_top = np.full(n, NAN)
    bull_bot = np.full(n, NAN)
    bear_top = np.full(n, NAN)
    bear_bot = np.full(n, NAN)
    bt = NAN
    bb = NAN
    st = NAN
    sb = NAN
    for t in range(n):
        if not np.isnan(bb):
            if close[t] < bb:
                bt = NAN
                bb = NAN
            elif low[t] <= bt:
                touch_bull[t] = True
        if not np.isnan(st):
            if close[t] > st:
                st = NAN
                sb = NAN
            elif high[t] >= sb:
                touch_bear[t] = True
        if bull_break[t]:
            for k in range(t - 1, max(t - lookback, 0) - 1, -1):
                if close[k] < open_[k]:
                    bt = high[k]
                    bb = low[k]
                    new_bull[t] = True
                    break
        if bear_break[t]:
            for k in range(t - 1, max(t - lookback, 0) - 1, -1):
                if close[k] > open_[k]:
                    st = high[k]
                    sb = low[k]
                    new_bear[t] = True
                    break
        bull_top[t] = bt
        bull_bot[t] = bb
        bear_top[t] = st
        bear_bot[t] = sb
    return new_bull, new_bear, touch_bull, touch_bear, bull_top, bull_bot, bear_top, bear_bot


# --------------------------------------------------------------------------- divergence
@njit(cache=True)
def divergences(high, low, osc, left, right, min_dist, max_dist):  # type: ignore[no-untyped-def]
    """Regular and hidden divergences between price pivots and an oscillator, emitted at
    the confirmation bar of the second pivot.

    regular bull: price lower low, osc higher low
    hidden bull: price higher low, osc lower low
    regular bear: price higher high, osc lower high
    hidden bear: price lower high, osc higher high
    """
    n = high.shape[0]
    ph, pl = pivots(high, low, left, right)
    reg_bull = np.zeros(n, dtype=np.bool_)
    hid_bull = np.zeros(n, dtype=np.bool_)
    reg_bear = np.zeros(n, dtype=np.bool_)
    hid_bear = np.zeros(n, dtype=np.bool_)
    last_l = -1
    last_h = -1
    for i in range(n - right):
        c = i + right
        if pl[i] and not np.isnan(osc[i]):
            if last_l >= 0 and min_dist <= i - last_l <= max_dist:
                if low[i] < low[last_l] and osc[i] > osc[last_l]:
                    reg_bull[c] = True
                elif low[i] > low[last_l] and osc[i] < osc[last_l]:
                    hid_bull[c] = True
            last_l = i
        if ph[i] and not np.isnan(osc[i]):
            if last_h >= 0 and min_dist <= i - last_h <= max_dist:
                if high[i] > high[last_h] and osc[i] < osc[last_h]:
                    reg_bear[c] = True
                elif high[i] < high[last_h] and osc[i] > osc[last_h]:
                    hid_bear[c] = True
            last_h = i
    return reg_bull, hid_bull, reg_bear, hid_bear


@njit(cache=True)
def bars_since(event: np.ndarray) -> np.ndarray:
    """Bars since the last True (0 on the event bar), -1 before the first event."""
    out = np.full(event.shape[0], -1, dtype=np.int64)
    last = -1
    for t in range(event.shape[0]):
        if event[t]:
            last = t
        if last >= 0:
            out[t] = t - last
    return out
