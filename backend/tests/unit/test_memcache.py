from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from engine.memcache import MB, BoundedCache, default_workers, nbytes
from engine.strategy.compiler import FeatureProvider
from engine.strategy.genome import FeatureSpec


def test_lru_eviction_by_bytes() -> None:
    c = BoundedCache(max_bytes=3 * 800)
    for k in "abc":
        c[k] = np.zeros(100)  # 800 bytes each
    _ = c["a"]  # a is now most recently used
    c["d"] = np.zeros(100)
    assert "b" not in c and {"a", "c", "d"} <= {k for k in "abcd" if k in c}
    assert c.bytes <= c.max_bytes and c.evictions == 1
    c["big"] = np.zeros(10_000)  # larger than the budget: kept alone, the rest evicted
    assert len(c) == 1 and "big" in c


def test_get_or_compute_counts_and_recomputes_after_eviction() -> None:
    c = BoundedCache(max_bytes=900)
    calls: list[str] = []

    def make(k: str) -> np.ndarray:
        calls.append(k)
        return np.full(100, ord(k), dtype=np.float64)

    a1 = c.get_or_compute("a", lambda: make("a"))
    assert c.get_or_compute("a", lambda: make("a")) is a1 and c.hits == 1
    c.get_or_compute("b", lambda: make("b"))  # evicts a (budget fits one array)
    a2 = c.get_or_compute("a", lambda: make("a"))
    assert calls == ["a", "b", "a"] and np.array_equal(a1, a2)


def test_nbytes_nested() -> None:
    value = {"x": np.zeros(10), "y": (np.zeros(5), pd.Series(np.zeros(4)))}
    assert nbytes(value) >= 80 + 40 + 32


def test_default_workers_respects_cpu_and_ram() -> None:
    gb = 1024 * MB
    assert default_workers(cpu_count=16, available=64 * gb) == 15  # CPU-bound
    assert default_workers(cpu_count=16, available=4 * gb) == 3  # RAM-bound
    assert default_workers(cpu_count=16, available=100 * MB) == 1  # never zero
    assert default_workers(cpu_count=1, available=64 * gb) == 1


def test_feature_cache_stays_bounded_over_many_parameter_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The swarm's access pattern: hundreds of distinct indicator parameterisations."""
    monkeypatch.setenv("SWARM_FEATURE_CACHE_MB", "16")
    n = 50_000
    rng = np.random.default_rng(1)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    ts = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "ts": ts,
            **{f"bid_{f}": close for f in ("open", "close")},
            "bid_high": close + 0.5,
            "bid_low": close - 0.5,
            **{f"ask_{f}": close + 0.3 for f in ("open", "close")},
            "ask_high": close + 0.8,
            "ask_low": close - 0.2,
            "volume": np.ones(n),
            "tick_count": np.ones(n),
        }
    )
    fp = FeatureProvider(bars, Timeframe.M15, None)
    first = fp.get(FeatureSpec(indicator="rsi", params={"length": 5}, output="value")).copy()
    for length in range(2, 101):  # every allowed RSI and EMA length: ~200 arrays
        fp.get(FeatureSpec(indicator="rsi", params={"length": length}, output="value"))
        fp.get(FeatureSpec(indicator="ema", params={"length": length}, output="value"))
    cache = fp.inputs._cache
    assert cache.bytes <= cache.max_bytes and cache.evictions > 0
    again = fp.get(FeatureSpec(indicator="rsi", params={"length": 5}, output="value"))
    np.testing.assert_array_equal(first, again)  # recomputed after eviction: identical


def test_ordered_params_are_rejected_at_validation_time() -> None:
    """Perturbation must never produce e.g. regime trend_fast >= trend_slow (it used to
    crash validation at compute time)."""
    from pydantic import ValidationError

    from engine.strategy.genome import Genome
    from engine.strategy.mutate import perturb

    with pytest.raises(ValidationError, match="trend_fast must be < trend_slow"):
        FeatureSpec(
            indicator="regime",
            params={"trend_fast": 150, "trend_slow": 120},
            output="trend_regime",
        )
    from engine.strategy.templates import TEMPLATES

    tpl = dict(next(iter(TEMPLATES.values())))
    regime = {
        "type": "state",
        "feature": {
            "indicator": "regime",
            "output": "trend_regime",
            "params": {"trend_fast": 195, "trend_slow": 200},
        },
        "values": [1],
    }
    tpl["filter"] = {"type": "group", "op": "all", "conditions": [regime]}
    genome = Genome.model_validate(tpl)
    rng = np.random.default_rng(0)
    for _ in range(200):
        variant = perturb(genome, rng, scale=0.3).canonical()
        params = variant["filter"]["conditions"][0]["feature"]["params"]
        assert params.get("trend_fast", 50) < params.get("trend_slow", 200)


def test_random_features_respect_ordered_params() -> None:
    from engine.evolution.operators import _feat
    from engine.indicators.base import get_spec

    rng = np.random.default_rng(3)
    for name, output in (("macd", "hist"), ("regime", "trend_regime"), ("divergence", "bull")):
        spec = get_spec(name)
        if output not in spec.outputs:
            output = next(iter(spec.outputs))
        for _ in range(500):
            feat = _feat(name, output, rng)
            spec.resolve(feat["params"])  # raises if an ordered pair is crossed
