"""Plain-English rationale for a live signal: which conditions fired (with values), the
context rule, the exit plan, and the champion's track record."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.strategy.compiler import FeatureProvider, evaluate
from engine.strategy.explain import condition_text
from engine.strategy.genome import Compare, Cross, FeatureOperand, Genome, Group


def _value(cond: Any, fp: FeatureProvider, t: int) -> float | None:
    if isinstance(cond, (Compare, Cross)) and isinstance(cond.left, FeatureOperand):
        v = float(np.asarray(fp.get(cond.left.feature), dtype=float)[t])
        return None if np.isnan(v) else round(v, 4)
    return None


def conditions_at(group: Group | None, fp: FeatureProvider, t: int) -> list[dict[str, Any]]:
    if group is None:
        return []
    out = []
    for cond in group.conditions:
        out.append(
            {
                "label": condition_text(cond),
                "met": bool(evaluate(cond, fp)[t]),
                "value": _value(cond, fp, t),
            }
        )
    return out


def build_rationale(
    genome: Genome, side: str, fp: FeatureProvider, t: int, meta: dict[str, Any]
) -> dict[str, Any]:
    entry = genome.entry_long if side == "long" else genome.entry_short
    return {
        "conditions": conditions_at(entry, fp, t),
        "conditions_op": entry.op if entry is not None else "all",  # "any": one met is enough
        "filters": conditions_at(genome.filter, fp, t),
        "filters_op": genome.filter.op if genome.filter is not None else "all",
        "context_rule": meta.get("context_rule"),
        "champion": meta.get("stats", {}),
        "strategy": meta.get("name"),
    }


def rationale_text(
    symbol: str,
    timeframe: str,
    side: str,
    price: float,
    stop: float | None,
    target: float | None,
    rationale: dict[str, Any],
) -> str:
    met = [c for c in rationale["conditions"] + rationale["filters"] if c["met"]]
    why = ", ".join(
        c["label"] + (f" (={c['value']:g})" if c["value"] is not None else "") for c in met
    )
    parts = [f"{side.upper()} {symbol} {timeframe} @ ~{price:.2f}", f"Why: {why}"]
    if rationale.get("context_rule"):
        parts.append(f"Context: {rationale['context_rule']}")
    plan = []
    if stop is not None:
        plan.append(f"stop {stop:.2f}")
    if target is not None:
        plan.append(f"target {target:.2f}")
    if plan:
        parts.append(" / ".join(plan))
    st = rationale.get("champion") or {}
    if st.get("oos_expectancy_r") is not None:
        parts.append(
            f"Track record: OOS {st['oos_expectancy_r']:+.2f}R/trade "
            f"over {st.get('oos_trades')} trades"
            + (
                f", holdout {st['holdout_expectancy_r']:+.2f}R"
                if st.get("holdout_expectancy_r") is not None
                else ""
            )
        )
    return " | ".join(parts)
