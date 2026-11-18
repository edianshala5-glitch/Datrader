"""Swarm: NSGA-II, genetic operators, fitness, engine determinism/checkpointing, and the
end-to-end controls (edge-free data → nothing validated; planted edge → rediscovered)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.models.enums import Timeframe
from app.services.seed import XAUUSD
from engine.backtest.runner import CostConfig
from engine.data.calendar import MarketCalendar
from engine.data.resample import resample_bars
from engine.data.sources.synthetic import generate_m1
from engine.evolution import nsga2
from engine.evolution.engine import EvolutionConfig, EvolutionEngine
from engine.evolution.evaluator import EvalContext, Evaluator
from engine.evolution.fitness import FLOOR, FitnessConfig
from engine.evolution.operators import crossover, decompose, mirror, mutate, random_genome
from engine.strategy.genome import MAX_CONDITIONS, Genome, Group, walk
from engine.strategy.templates import f, feat, group
from engine.validation.splits import make_splits
from engine.validation.validator import validate_genome

CAL = MarketCalendar.xauusd()
SESSIONS = XAUUSD.model_dump(mode="json")["sessions"]
COSTS = CostConfig.from_instrument(XAUUSD)
SPLITS = make_splits("2021-01-03", "2024-01-01")
GA_END = SPLITS.folds[0].train_end


# --------------------------------------------------------------------------- NSGA-II
def test_fronts_crowding_and_selection() -> None:
    obj = np.array([[1, 5], [2, 4], [3, 3], [2, 2], [1, 1], [4, 1]], float)
    assert nsga2.non_dominated_fronts(obj) == [[0, 1, 2, 5], [3], [4]]
    d = nsga2.crowding_distance(obj, [0, 1, 2, 5])
    assert d[0] == d[5] == float("inf") and 0 < d[1] < float("inf")
    assert set(nsga2.select(obj, 4)) == {0, 1, 2, 5}
    picked = nsga2.select(obj, 3)
    assert 0 in picked and 5 in picked  # extremes of the front survive
    rank, crowd = nsga2.rank_and_crowding(obj)
    rng = np.random.default_rng(0)
    wins = [nsga2.tournament(rank, crowd, rng, size=6) for _ in range(20)]
    assert all(rank[w] == 0 for w in wins)


# --------------------------------------------------------------------------- operators
def test_random_genomes_are_valid_unique_and_within_budget() -> None:
    rng = np.random.default_rng(0)
    gs = [random_genome(rng, htf=(Timeframe.H1,)) for _ in range(300)]
    assert len({g.model_dump_json() for g in gs}) == 300
    for g in gs:
        n = sum(
            1 for r in g.condition_roots() for node, _ in walk(r) if not isinstance(node, Group)
        )
        assert n <= MAX_CONDITIONS
    assert {decompose(g)[3] for g in gs} == {"both", "long", "short"}


def test_mirror_semantics() -> None:
    long_side = group(
        {"type": "event", "feature": f("structure", "choch_bull"), "within": 5},
        {
            "type": "compare",
            "left": feat("rsi", "value"),
            "op": "<",
            "right": {"kind": "const", "value": 30},
        },
        {
            "type": "compare",
            "left": {"kind": "price", "field": "close"},
            "op": ">",
            "right": feat("ema", "value"),
        },
        {
            "type": "compare",
            "left": feat("bbands", "pct_b"),
            "op": "<",
            "right": {"kind": "const", "value": 0.2},
        },
        {"type": "state", "feature": f("structure", "trend"), "values": [1]},
        {"type": "flag", "feature": f("sessions", "ny_am_kz")},
        {"type": "range", "feature": f("time", "minute_ny"), "lo": 570, "hi": 660},
        {
            "type": "compare",
            "left": feat("adx", "adx"),
            "op": ">",
            "right": {"kind": "const", "value": 25},
        },
    )
    m = mirror(long_side)["conditions"]
    assert m[0]["feature"]["output"] == "choch_bear" and m[0]["within"] == 5
    assert (m[1]["op"], m[1]["right"]["value"]) == (">", 70)
    assert m[2]["op"] == "<"
    assert (m[3]["op"], m[3]["right"]["value"]) == (">", pytest.approx(0.8))
    assert m[4]["values"] == [-1]
    assert m[5] == long_side["conditions"][5]  # sessions are direction-neutral
    assert m[6] == long_side["conditions"][6]  # so is the clock
    assert m[7] == long_side["conditions"][7]  # and trend *strength*
    assert mirror(mirror(long_side)) == long_side
    Genome.model_validate({"entry_short": mirror(long_side)})  # mirrored side is valid


def test_mutation_and_crossover_produce_valid_children() -> None:
    rng = np.random.default_rng(1)
    parents = [random_genome(rng) for _ in range(60)]
    for a, b in zip(parents[::2], parents[1::2], strict=True):
        child = crossover(a, b, rng)
        mutant = mutate(child, rng, max_entry=5)
        for g in (child, mutant):
            Genome.model_validate(g.canonical())
        assert mutant != child


def test_irrelevant_params_are_stripped() -> None:
    g = Genome.model_validate(
        {
            "entry_long": group(
                {
                    "type": "state",
                    "feature": f("regime", "vol_regime", trend_fast=123, window=2500),
                    "values": [1],
                },
                {"type": "event", "feature": f("structure", "bos_bull", eq_tol=0.7, length=8)},
            )
        }
    )
    params = [c.feature.params for c in g.entry_long.conditions]  # type: ignore[union-attr]
    assert params == [{"window": 2500}, {"length": 8}]


# --------------------------------------------------------------------------- engine
def _bars(mu: float, seed: int) -> pd.DataFrame:
    def drift(m: pd.DatetimeIndex) -> np.ndarray:  # type: ignore[type-arg]
        h = m.tz_convert("America/New_York").hour
        return np.where((h >= 8) & (h < 11), mu, 0.0)

    m1 = generate_m1(
        datetime(2021, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
        CAL,
        seed=seed,
        drift=drift,
    )
    return resample_bars(m1, Timeframe.M15, CAL)


@pytest.fixture(scope="module")
def flat() -> pd.DataFrame:
    return _bars(0.0, 9)


def _ctx(bars: pd.DataFrame, min_trades: int = 30) -> EvalContext:
    return EvalContext(
        bars[bars["ts"] < GA_END].reset_index(drop=True),
        Timeframe.M15,
        SPLITS.research_start,
        GA_END,
        CAL,
        SESSIONS,
        COSTS,
        FitnessConfig(min_trades=min_trades),
    )


SMALL = EvolutionConfig(islands=2, population=12, generations=4, migration_interval=2, seed=7)


def test_engine_is_deterministic_and_dedupes(flat: pd.DataFrame) -> None:
    runs = []
    for _ in range(2):
        ev = Evaluator(_ctx(flat))
        eng = EvolutionEngine(SMALL, ev)
        reports = []
        eng.run(reports.append)
        runs.append(([i.key for i in eng.best(5)], ev.evaluations, reports))
    assert runs[0][:2] == runs[1][:2]
    _keys, evals, reports = runs[0]
    assert [r.number for r in reports] == [0, 1, 2, 3, 4]
    assert evals == len(ev.cache) == len(eng._seen)  # every unique genome counted once
    assert any(r.migrated for r in reports)
    assert all(len(pop) == SMALL.population for pop in eng.islands)
    assert all(i.fitness is not None for pop in eng.islands for i in pop)


def test_parallel_matches_serial(flat: pd.DataFrame) -> None:
    serial = EvolutionEngine(SMALL, Evaluator(_ctx(flat)))
    serial.run()
    with Evaluator(_ctx(flat), workers=2) as pev:
        parallel = EvolutionEngine(SMALL, pev)
        parallel.run()
        assert [i.key for i in parallel.best(5)] == [i.key for i in serial.best(5)]
        assert [round(i.score, 9) for i in parallel.best(5)] == [
            round(i.score, 9) for i in serial.best(5)
        ]


def test_checkpoint_resume_equals_uninterrupted(flat: pd.DataFrame) -> None:
    straight = EvolutionEngine(SMALL, Evaluator(_ctx(flat)))
    straight.run()

    first = EvolutionEngine(SMALL, Evaluator(_ctx(flat)))
    first.step()
    first.step()  # generations 0 and 1, then "crash"
    state = json.loads(json.dumps(first.state_dict(), default=str))  # through real JSON
    resumed = EvolutionEngine.from_state(state, Evaluator(_ctx(flat)))
    resumed.run()
    assert resumed.generation == straight.generation
    assert resumed.evaluator.evaluations == straight.evaluator.evaluations
    assert [i.key for i in resumed.best(10)] == [i.key for i in straight.best(10)]


def test_fitness_floor_for_inactive_genomes(flat: pd.DataFrame) -> None:
    never = Genome.model_validate(
        {
            "entry_long": group(
                {
                    "type": "compare",
                    "left": feat("rsi", "value"),
                    "op": ">",
                    "right": {"kind": "const", "value": 100},
                }
            )
        }
    )
    [fit] = Evaluator(_ctx(flat)).evaluate([never])
    assert fit.objectives == (FLOOR, FLOOR, FLOOR) and not fit.valid


# --------------------------------------------------------------------------- end-to-end controls
def _evolve_and_validate(bars: pd.DataFrame, seed: int, k: int):  # type: ignore[no-untyped-def]
    cfg = EvolutionConfig(islands=4, population=30, generations=12, seed=seed)
    eng = EvolutionEngine(cfg, Evaluator(_ctx(bars)))
    eng.run()
    return [
        (
            ind,
            validate_genome(
                ind.genome,
                bars,
                Timeframe.M15,
                SPLITS,
                calendar=CAL,
                sessions=SESSIONS,
                costs=COSTS,
                n_trials=k,
            ),
        )
        for ind in eng.best(k)
    ]


def test_swarm_validates_nothing_on_edge_free_data(flat: pd.DataFrame) -> None:
    results = _evolve_and_validate(flat, seed=1, k=10)
    assert results and not any(rep.passed for _, rep in results)
    assert min(rep.overfit_risk for _, rep in results) > 0.4


def test_swarm_rediscovers_planted_edge() -> None:
    """Regression test for a known-good configuration. Discovery at this tiny budget isn't
    guaranteed for every seed (see docs/VALIDATION.md); certification must only ever
    certify the planted morning edge."""
    bars = _bars(6e-6, 9)
    # At this budget discovery happens for roughly 3-4 seeds in 8 (measured), so try a few
    # instead of pinning one: a change to the random generator shifts which seeds succeed.
    reasons = []
    for seed in (0, 1, 2, 3):
        results = _evolve_and_validate(bars, seed=seed, k=10)
        passed = [(ind, rep) for ind, rep in results if rep.passed]
        if passed:
            break
        reasons.append([rep.reasons for _, rep in results])
    assert passed, reasons
    for ind, rep in passed:
        assert rep.oos_metrics["expectancy_r"] > 0.05
        ny_hours = set()
        for root in ind.genome.condition_roots():
            for node, _ in walk(root):
                feat_ = getattr(node, "feature", None)
                if feat_ is not None and feat_.indicator in ("time", "sessions"):
                    ny_hours.add(feat_.output)
        assert ny_hours, "a validated strategy should exploit the time-of-day edge"
