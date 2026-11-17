"""The no-lookahead contract, enforced for every registered indicator.

For each indicator and several parameter sets, values computed on the first ``k`` bars must
equal values computed on the full history, for every bar ``< k``. Any dependence on future
bars (centered windows, unconfirmed pivots, backfilled values) breaks this equality.
New indicators are covered automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.indicators import REGISTRY, Inputs, compute

CAL = MarketCalendar.xauusd()

# Extra parameter sets beyond defaults, exercising other code paths.
VARIANTS: dict[str, list[dict[str, Any]]] = {
    "structure": [{"length": 3, "break_on": "wick"}, {"length": 8, "eq_tol": 0.5}],
    "divergence": [{"osc": "macd_hist", "length": 3}],
    "fvg": [{"min_size_atr": 0.0}, {"min_size_atr": 0.5}],
    "order_blocks": [{"length": 3, "lookback": 5}],
    "swings": [{"length": 2}],
    "regime": [{"window": 500, "trend_fast": 20, "trend_slow": 50}],
}


@pytest.fixture(scope="module")
def full() -> Inputs:
    m1 = generate_m1(
        datetime(2024, 1, 7, tzinfo=UTC), datetime(2024, 2, 18, tzinfo=UTC), CAL, seed=11
    )
    bars = resample_bars(m1, Timeframe.M5, CAL)
    return Inputs.from_bars(bars, Timeframe.M5, CAL, XAUUSD.model_dump(mode="json")["sessions"])


CASES = [
    pytest.param(name, params, id=f"{name}-{i}")
    for name in sorted(REGISTRY)
    for i, params in enumerate([{}, *VARIANTS.get(name, [])])
]


def assert_causal(full: Inputs, fn: Callable[[Inputs], dict[str, np.ndarray]], name: str) -> None:  # type: ignore[type-arg]
    n = len(full)
    reference = fn(full)
    for cut in (300, 1777, n // 2, n - 13):
        truncated = fn(full.slice(cut))
        for out, values in reference.items():
            head = values[:cut]
            got = truncated[out]
            assert got.shape == head.shape, f"{name}.{out}"
            if np.issubdtype(head.dtype, np.floating):
                np.testing.assert_allclose(
                    got, head, rtol=1e-12, atol=0, equal_nan=True, err_msg=f"{name}.{out} cut={cut}"
                )
            else:
                np.testing.assert_array_equal(got, head, err_msg=f"{name}.{out} cut={cut}")


@pytest.mark.parametrize(("name", "params"), CASES)
def test_no_lookahead(full: Inputs, name: str, params: dict[str, Any]) -> None:
    assert len(full) > 5000
    assert_causal(full, lambda inp: compute(inp, name, params), name)


def test_harness_catches_leaks(full: Inputs) -> None:
    """The check itself must fail on classic leaks, or a green run would mean nothing."""
    from engine.indicators.kernels import pivots

    def centered_ma(inp: Inputs) -> dict[str, np.ndarray]:  # type: ignore[type-arg]
        return {"v": pd.Series(inp.close).rolling(5, center=True).mean().to_numpy()}

    def pivots_at_pivot_bar(inp: Inputs) -> dict[str, np.ndarray]:  # type: ignore[type-arg]
        ph, _ = pivots(inp.high, inp.low, 5, 5)  # not shifted to confirmation time
        return {"ph": ph}

    for leaky in (centered_ma, pivots_at_pivot_bar):
        with pytest.raises(AssertionError):
            assert_causal(full, leaky, leaky.__name__)


def test_every_indicator_is_covered() -> None:
    assert {c.values[0] for c in CASES} == set(REGISTRY)
    assert len(REGISTRY) >= 19


def test_outputs_have_declared_length_and_events_are_bool(full: Inputs) -> None:
    from engine.indicators import OutputKind

    for name, spec in REGISTRY.items():
        outs = compute(full, name)
        for out, decl in spec.outputs.items():
            assert len(outs[out]) == len(full), f"{name}.{out}"
            if decl.kind in (OutputKind.EVENT, OutputKind.FLAG):
                assert outs[out].dtype == np.bool_, f"{name}.{out} should be bool"
            if decl.kind is OutputKind.STATE and decl.values:
                assert set(np.unique(outs[out])) <= set(decl.values), f"{name}.{out}"
