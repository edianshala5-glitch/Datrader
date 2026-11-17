"""Island-model NSGA-II over strategy genomes.

Each island evolves its own population. Every ``migration_interval`` generations the best
individuals of island *i* replace the worst of island *i+1* (ring topology). Isolated
islands explore different regions of strategy space, and migration shares the discoveries.

Determinism: one seeded RNG drives every random choice in a fixed order, and evaluation
results return in input order. The same config and seed give the same run, serial or
parallel, and :meth:`state_dict` / :meth:`from_state` resume exactly where a run stopped.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.models.enums import StrategyOrigin, Timeframe
from engine.evolution.evaluator import Evaluator
from engine.evolution.fitness import FLOOR, Fitness, FitnessConfig
from engine.evolution.nsga2 import non_dominated_fronts, rank_and_crowding, select, tournament
from engine.evolution.operators import crossover, mutate, random_genome
from engine.models_util import genome_key
from engine.strategy.genome import Genome
from engine.strategy.templates import all_templates


@dataclass(frozen=True)
class EvolutionConfig:
    islands: int = 4
    population: int = 40
    generations: int = 25
    crossover_rate: float = 0.5
    mutation_rate: float = 0.9
    tournament_size: int = 2
    migration_interval: int = 5
    migrants: int = 2
    random_immigrants: float = 0.05
    seed_templates: bool = True
    max_entry: int = 5
    htf: tuple[str, ...] = ()
    hall_of_fame: int = 50
    seed: int = 0
    fitness: FitnessConfig = field(default_factory=FitnessConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["htf"] = list(self.htf)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionConfig:
        d = dict(d)
        d["fitness"] = FitnessConfig(**d.get("fitness", {}))
        d["htf"] = tuple(d.get("htf", ()))
        return cls(**d)

    @property
    def htf_timeframes(self) -> tuple[Timeframe, ...]:
        return tuple(Timeframe(t) for t in self.htf)


@dataclass
class Individual:
    genome: Genome
    key: str
    origin: StrategyOrigin
    parents: tuple[str, ...]
    island: int
    born: int
    fitness: Fitness | None = None

    @property
    def score(self) -> float:
        return self.fitness.fitness if self.fitness else FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome": self.genome.canonical(),
            "key": self.key,
            "origin": self.origin.value,
            "parents": list(self.parents),
            "island": self.island,
            "born": self.born,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], fitness: Fitness | None) -> Individual:
        return cls(
            Genome.model_validate(d["genome"]),
            d["key"],
            StrategyOrigin(d["origin"]),
            tuple(d["parents"]),
            d["island"],
            d["born"],
            fitness,
        )


@dataclass
class GenerationReport:
    number: int
    islands: list[dict[str, Any]]
    new: list[Individual]  # first-time evaluated this generation (for persistence)
    evaluations: int  # unique genomes evaluated so far in the run
    best: Individual | None
    duration_ms: int
    migrated: bool = False


def _island_stats(pop: list[Individual]) -> dict[str, Any]:
    valid = [i for i in pop if i.fitness and i.fitness.valid]
    fits = np.array([i.score for i in valid]) if valid else np.array([])
    obj = (
        np.array([i.fitness.objectives for i in valid if i.fitness]) if valid else np.zeros((0, 3))
    )
    best = max(valid, key=lambda i: i.score) if valid else None
    return {
        "size": len(pop),
        "valid": len(valid),
        "diversity": round(len({i.key for i in pop}) / max(len(pop), 1), 4),
        "fitness": {"min": float(fits.min()), "mean": float(fits.mean()), "max": float(fits.max())}
        if len(fits)
        else None,
        "edge_max": float(obj[:, 0].max()) if len(obj) else None,
        "pareto_size": len(non_dominated_fronts(obj)[0]) if len(obj) else 0,
        "best_key": best.key if best else None,
        "best_metrics": best.fitness.metrics if best and best.fitness else None,
    }


class EvolutionEngine:
    def __init__(self, config: EvolutionConfig, evaluator: Evaluator) -> None:
        self.cfg = config
        self.evaluator = evaluator
        self.rng = np.random.default_rng(config.seed)
        self.islands: list[list[Individual]] = []
        self.generation = -1
        self.hall: dict[str, Individual] = {}
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _new(
        self, genome: Genome, origin: StrategyOrigin, parents: tuple[str, ...], island: int
    ) -> Individual:
        return Individual(genome, genome_key(genome), origin, parents, island, self.generation)

    def _evaluate(self, inds: list[Individual]) -> list[Individual]:
        """Score individuals; return those evaluated for the first time in this run."""
        fits = self.evaluator.evaluate([i.genome for i in inds])
        fresh = []
        for ind, f in zip(inds, fits, strict=True):
            ind.fitness = f
            if ind.key not in self._seen:
                self._seen.add(ind.key)
                fresh.append(ind)
            if f.valid and (ind.key not in self.hall or self.hall[ind.key].score < f.fitness):
                self.hall[ind.key] = ind
        if len(self.hall) > self.cfg.hall_of_fame * 2:
            keep = sorted(self.hall.values(), key=lambda i: -i.score)[: self.cfg.hall_of_fame]
            self.hall = {i.key: i for i in keep}
        return fresh

    def _report(self, new: list[Individual], t0: float, migrated: bool = False) -> GenerationReport:
        best = max(self.hall.values(), key=lambda i: i.score) if self.hall else None
        return GenerationReport(
            number=self.generation,
            islands=[_island_stats(pop) for pop in self.islands],
            new=new,
            evaluations=self.evaluator.evaluations,
            best=best,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            migrated=migrated,
        )

    # ------------------------------------------------------------------ lifecycle
    def initialize(self) -> GenerationReport:
        t0 = time.perf_counter()
        self.generation = 0
        htf = self.cfg.htf_timeframes
        templates = list(all_templates().values()) if self.cfg.seed_templates else []
        self.islands = []
        for isl in range(self.cfg.islands):
            pop: list[Individual] = []
            keys: set[str] = set()
            if templates:
                t = templates[isl % len(templates)]  # each island starts near a different idea
                for g in [t] + [
                    mutate(t, self.rng, htf=htf) for _ in range(max(1, self.cfg.population // 8))
                ]:
                    ind = self._new(g, StrategyOrigin.SEED, (), isl)
                    if ind.key not in keys:
                        keys.add(ind.key)
                        pop.append(ind)
            while len(pop) < self.cfg.population:
                ind = self._new(random_genome(self.rng, htf=htf), StrategyOrigin.RANDOM, (), isl)
                if ind.key not in keys:
                    keys.add(ind.key)
                    pop.append(ind)
            self.islands.append(pop)
        new = self._evaluate([i for pop in self.islands for i in pop])
        return self._report(new, t0)

    def _offspring(self, pop: list[Individual], island: int) -> list[Individual]:
        cfg, htf = self.cfg, self.cfg.htf_timeframes
        obj = np.array([i.fitness.objectives if i.fitness else (FLOOR,) * 3 for i in pop])
        rank, crowd = rank_and_crowding(obj)
        taken = {i.key for i in pop}
        kids: list[Individual] = []
        attempts = 0
        while len(kids) < cfg.population and attempts < cfg.population * 10:
            attempts += 1
            if self.rng.random() < cfg.random_immigrants:
                child = self._new(
                    random_genome(self.rng, htf=htf), StrategyOrigin.RANDOM, (), island
                )
            else:
                p1 = pop[tournament(rank, crowd, self.rng, cfg.tournament_size)]
                parents: tuple[str, ...]
                if self.rng.random() < cfg.crossover_rate:
                    p2 = pop[tournament(rank, crowd, self.rng, cfg.tournament_size)]
                    g = crossover(p1.genome, p2.genome, self.rng)
                    origin, parents = StrategyOrigin.CROSSOVER, (p1.key, p2.key)
                    if self.rng.random() < cfg.mutation_rate:
                        g = mutate(g, self.rng, htf=htf, max_entry=cfg.max_entry)
                else:
                    g = mutate(p1.genome, self.rng, htf=htf, max_entry=cfg.max_entry)
                    origin, parents = StrategyOrigin.MUTATION, (p1.key,)
                child = self._new(g, origin, tuple(dict.fromkeys(parents)), island)
            if child.key in taken:
                continue
            taken.add(child.key)
            kids.append(child)
        return kids

    def step(self) -> GenerationReport:
        if self.generation < 0:
            return self.initialize()
        t0 = time.perf_counter()
        self.generation += 1
        offspring = [self._offspring(pop, isl) for isl, pop in enumerate(self.islands)]
        new = self._evaluate([k for kids in offspring for k in kids])
        for isl, (pop, kids) in enumerate(zip(self.islands, offspring, strict=True)):
            combined = pop + kids
            obj = np.array([i.fitness.objectives for i in combined])  # type: ignore[union-attr]
            self.islands[isl] = [combined[j] for j in select(obj, self.cfg.population)]
        migrated = False
        n = len(self.islands)
        if (
            n > 1
            and self.cfg.migration_interval
            and self.generation % self.cfg.migration_interval == 0
        ):
            migrated = True
            emigrants = [
                sorted(pop, key=lambda i: -i.score)[: self.cfg.migrants] for pop in self.islands
            ]
            for isl in range(n):
                dest = self.islands[(isl + 1) % n]
                present = {i.key for i in dest}
                incoming = [m for m in emigrants[isl] if m.key not in present]
                if not incoming:
                    continue
                dest.sort(key=lambda i: i.score)
                for j, m in enumerate(incoming):
                    dest[j] = Individual(
                        m.genome, m.key, m.origin, m.parents, (isl + 1) % n, m.born, m.fitness
                    )
        return self._report(new, t0, migrated)

    def run(
        self,
        on_generation: Callable[[GenerationReport], None] | None = None,
        stop: threading.Event | None = None,
    ) -> GenerationReport:
        report = self.step() if self.generation < 0 else None
        if report and on_generation:
            on_generation(report)
        while self.generation < self.cfg.generations:
            if stop is not None and stop.is_set():
                break
            report = self.step()
            if on_generation:
                on_generation(report)
        assert report is not None or self.generation >= 0
        return report if report is not None else self._report([], time.perf_counter())

    def best(self, k: int = 10) -> list[Individual]:
        return sorted(self.hall.values(), key=lambda i: -i.score)[:k]

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict[str, Any]:
        def fit(f: Fitness) -> dict[str, Any]:
            return {"objectives": list(f.objectives), "fitness": f.fitness, "metrics": f.metrics}

        return {
            "version": 1,
            "config": self.cfg.to_dict(),
            "generation": self.generation,
            "rng": self.rng.bit_generator.state,
            "islands": [[i.to_dict() for i in pop] for pop in self.islands],
            "hall": [i.to_dict() for i in self.hall.values()],
            "seen": sorted(self._seen),
            "evaluations": self.evaluator.evaluations,
            "cache": {k: fit(f) for k, f in self.evaluator.cache.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], evaluator: Evaluator) -> EvolutionEngine:
        eng = cls(EvolutionConfig.from_dict(state["config"]), evaluator)
        eng.rng.bit_generator.state = state["rng"]
        eng.generation = state["generation"]
        evaluator.cache = {
            k: Fitness(
                (v["objectives"][0], v["objectives"][1], v["objectives"][2]),
                v["fitness"],
                v["metrics"],
            )
            for k, v in state["cache"].items()
        }
        evaluator.evaluations = state["evaluations"]
        eng.islands = [
            [Individual.from_dict(d, evaluator.cache.get(d["key"])) for d in pop]
            for pop in state["islands"]
        ]
        eng.hall = {
            d["key"]: Individual.from_dict(d, evaluator.cache.get(d["key"])) for d in state["hall"]
        }
        eng._seen = set(state["seen"])
        return eng
