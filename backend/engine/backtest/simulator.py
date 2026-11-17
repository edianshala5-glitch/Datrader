"""Bar-by-bar execution simulator (numba).

Execution model, conservative by design:

* Signals are evaluated at the close of bar ``t-1``. Entries fill at the **open of bar t**:
  longs at the ask, shorts at the bid, plus slippage.
* Stops are checked against the **bid low** (longs) / **ask high** (shorts). A gap through
  the stop at the open fills at the open, not at the stop.
* A bar that touches both stop and target counts as a **stop** (the order of the
  intrabar path is unknown, so assume the worst).
* Targets are limit orders: they fill at the target price with no slippage, or at a better
  open on a gap.
* Signal / time exits decided at a close fill at the next open. Forced flats before the
  daily break or weekend fill at the **close** of the last bar.
* Trailing stops are updated after a bar is processed and apply from the next bar.
* Commission is charged per lot round-turn at exit. Spread comes from the bid/ask data
  (with a floor applied by the caller).
* Size: ``risk_frac`` of current equity at risk between fill and stop, rounded down to the
  lot step. Trades smaller than the minimum lot are skipped.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# exit reason codes ↔ app.models.enums.ExitReason
STOP, TARGET, TRAILING, SIGNAL, TIME, SESSION, EOD = 0, 1, 2, 3, 4, 5, 6
EXIT_REASONS = ("stop", "target", "trailing_stop", "signal", "time", "session_close", "end_of_data")

TRADE_FIELDS = (
    "signal_idx",
    "entry_idx",
    "exit_idx",
    "side",
    "entry_price",
    "exit_price",
    "stop_price",
    "target_price",
    "lots",
    "pnl",
    "r",
    "mae_r",
    "mfe_r",
    "reason",
    "commission",
)


@njit(cache=True)
def simulate(  # type: ignore[no-untyped-def]
    bid_o,
    bid_h,
    bid_l,
    bid_c,
    ask_o,
    ask_h,
    ask_l,
    ask_c,
    entry_long,
    entry_short,
    exit_long,
    exit_short,
    stop_lvl_l,
    stop_lvl_s,
    stop_dist_l,
    stop_dist_s,
    tgt_dist_l,
    tgt_dist_s,
    target_rr,
    trail_dist,
    trail_activate_r,
    max_bars,
    flat_at,
    initial_equity,
    risk_frac,
    contract_size,
    lot_step,
    min_lot,
    max_lots,
    commission_rt,
    slippage,
):
    n = bid_o.shape[0]
    cap = n // 2 + 1
    tr = np.full((cap, 15), np.nan)
    k = 0
    equity_curve = np.empty(n)
    position = np.zeros(n, dtype=np.int8)

    realized = initial_equity
    pos = 0
    entry = 0.0
    stop = 0.0
    stop0 = 0.0
    target = np.nan
    lots = 0.0
    risk_px = 0.0
    e_idx = -1
    mfe = 0.0
    mae = 0.0
    trailed = False
    pending = -1  # reason code for an exit at this bar's open

    for t in range(n):
        # ---- 1) exits scheduled at the previous close fill at this open
        if pos != 0 and pending >= 0:
            px = bid_o[t] - slippage if pos == 1 else ask_o[t] + slippage
            comm = commission_rt * lots
            pnl = pos * (px - entry) * lots * contract_size - comm
            realized += pnl
            tr[k, 0] = e_idx - 1
            tr[k, 1] = e_idx
            tr[k, 2] = t
            tr[k, 3] = pos
            tr[k, 4] = entry
            tr[k, 5] = px
            tr[k, 6] = stop0
            tr[k, 7] = target
            tr[k, 8] = lots
            tr[k, 9] = pnl
            tr[k, 10] = pnl / (risk_px * lots * contract_size)
            tr[k, 11] = mae / risk_px
            tr[k, 12] = mfe / risk_px
            tr[k, 13] = pending
            tr[k, 14] = comm
            k += 1
            pos = 0
            pending = -1

        # ---- 2) entries from the previous close
        if pos == 0 and t >= 1 and not flat_at[t - 1] and not flat_at[t]:
            s = t - 1
            go_long = entry_long[s] and not entry_short[s]
            go_short = entry_short[s] and not entry_long[s]
            if go_long or go_short:
                side = 1 if go_long else -1
                if side == 1:
                    fill = ask_o[t] + slippage
                    st = stop_lvl_l[s] if not np.isnan(stop_lvl_l[s]) else fill - stop_dist_l[s]
                    risk = fill - st
                else:
                    fill = bid_o[t] - slippage
                    st = stop_lvl_s[s] if not np.isnan(stop_lvl_s[s]) else fill + stop_dist_s[s]
                    risk = st - fill
                if risk > 0 and np.isfinite(risk):
                    raw = realized * risk_frac / (risk * contract_size)
                    size = np.floor(raw / lot_step + 1e-9) * lot_step
                    if size > max_lots:
                        size = max_lots
                    if size >= min_lot:
                        pos = side
                        entry = fill
                        stop = st
                        stop0 = st
                        risk_px = risk
                        lots = size
                        e_idx = t
                        mfe = 0.0
                        mae = 0.0
                        trailed = False
                        if not np.isnan(target_rr):
                            target = fill + side * target_rr * risk
                        else:
                            d = tgt_dist_l[s] if side == 1 else tgt_dist_s[s]
                            target = fill + side * d if not np.isnan(d) else np.nan

        # ---- 3) manage the open position through bar t
        if pos != 0:
            exit_px = np.nan
            reason = -1
            if pos == 1:
                if t > e_idx and bid_o[t] <= stop:
                    exit_px = bid_o[t] - slippage
                    reason = TRAILING if trailed else STOP
                elif t > e_idx and not np.isnan(target) and bid_o[t] >= target:
                    exit_px = bid_o[t]
                    reason = TARGET
                elif bid_l[t] <= stop:
                    exit_px = stop - slippage
                    reason = TRAILING if trailed else STOP
                elif not np.isnan(target) and bid_h[t] >= target:
                    exit_px = target
                    reason = TARGET
            else:
                if t > e_idx and ask_o[t] >= stop:
                    exit_px = ask_o[t] + slippage
                    reason = TRAILING if trailed else STOP
                elif t > e_idx and not np.isnan(target) and ask_o[t] <= target:
                    exit_px = ask_o[t]
                    reason = TARGET
                elif ask_h[t] >= stop:
                    exit_px = stop + slippage
                    reason = TRAILING if trailed else STOP
                elif not np.isnan(target) and ask_l[t] <= target:
                    exit_px = target
                    reason = TARGET

            if reason < 0:
                # excursions (in price, favourable/adverse)
                fav = (bid_h[t] - entry) if pos == 1 else (entry - ask_l[t])
                adv = (entry - bid_l[t]) if pos == 1 else (ask_h[t] - entry)
                mfe = max(mfe, fav)
                mae = max(mae, adv)
                if not np.isnan(trail_dist[t]) and mfe >= trail_activate_r * risk_px:
                    if pos == 1:
                        cand = bid_h[t] - trail_dist[t]
                        if cand > stop:
                            stop = cand
                            trailed = True
                    else:
                        cand = ask_l[t] + trail_dist[t]
                        if cand < stop:
                            stop = cand
                            trailed = True
                if flat_at[t]:
                    exit_px = (bid_c[t] - slippage) if pos == 1 else (ask_c[t] + slippage)
                    reason = SESSION
                elif t == n - 1:
                    exit_px = bid_c[t] if pos == 1 else ask_c[t]
                    reason = EOD
                elif (exit_long[t] and pos == 1) or (exit_short[t] and pos == -1):
                    pending = SIGNAL
                elif max_bars > 0 and t - e_idx + 1 >= max_bars:
                    pending = TIME
            else:
                fav = (exit_px - entry) * pos
                mfe = max(mfe, fav)
                mae = max(mae, -fav)

            if reason >= 0:
                comm = commission_rt * lots
                pnl = pos * (exit_px - entry) * lots * contract_size - comm
                realized += pnl
                tr[k, 0] = e_idx - 1
                tr[k, 1] = e_idx
                tr[k, 2] = t
                tr[k, 3] = pos
                tr[k, 4] = entry
                tr[k, 5] = exit_px
                tr[k, 6] = stop0
                tr[k, 7] = target
                tr[k, 8] = lots
                tr[k, 9] = pnl
                tr[k, 10] = pnl / (risk_px * lots * contract_size)
                tr[k, 11] = mae / risk_px
                tr[k, 12] = mfe / risk_px
                tr[k, 13] = reason
                tr[k, 14] = comm
                k += 1
                pos = 0
                pending = -1

        # ---- 4) mark to market at the close
        if pos == 1:
            equity_curve[t] = realized + (bid_c[t] - entry) * lots * contract_size
        elif pos == -1:
            equity_curve[t] = realized + (entry - ask_c[t]) * lots * contract_size
        else:
            equity_curve[t] = realized
        position[t] = pos

    return tr[:k], equity_curve, position
