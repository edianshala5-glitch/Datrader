"""NSGA-II selection (Deb et al., 2002): fast non-dominated sorting + crowding distance.

All objectives are **maximized**. Individuals are ranked by Pareto front (0 = best), then
by crowding distance within a front (more isolated = preferred, preserving diversity).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Objectives = npt.NDArray[np.float64]  # shape (N, M)


def dominates(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_fronts(obj: Objectives) -> list[list[int]]:
    n = len(obj)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(p + 1, n):
            if dominates(obj[p], obj[q]):
                dominated_by[p].append(q)
                counts[q] += 1
            elif dominates(obj[q], obj[p]):
                dominated_by[q].append(p)
                counts[p] += 1
    fronts[0] = [i for i in range(n) if counts[i] == 0]
    k = 0
    while fronts[k]:
        nxt = []
        for p in fronts[k]:
            for q in dominated_by[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    nxt.append(q)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(obj: Objectives, front: list[int]) -> dict[int, float]:
    dist = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        return dict.fromkeys(front, float("inf"))
    sub = obj[front]
    for m in range(obj.shape[1]):
        order = np.argsort(sub[:, m], kind="stable")
        lo, hi = sub[order[0], m], sub[order[-1], m]
        dist[front[order[0]]] = dist[front[order[-1]]] = float("inf")
        if hi == lo:
            continue
        for j in range(1, len(front) - 1):
            dist[front[order[j]]] += (sub[order[j + 1], m] - sub[order[j - 1], m]) / (hi - lo)
    return dist


def rank_and_crowding(obj: Objectives) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    rank = np.zeros(len(obj), dtype=np.int64)
    crowd = np.zeros(len(obj))
    for r, front in enumerate(non_dominated_fronts(obj)):
        for i, d in crowding_distance(obj, front).items():
            rank[i] = r
            crowd[i] = d
    return rank, crowd


def select(obj: Objectives, k: int) -> list[int]:
    """Indices of the ``k`` survivors: whole fronts first, the last front cut by crowding."""
    chosen: list[int] = []
    for front in non_dominated_fronts(obj):
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
            continue
        dist = crowding_distance(obj, front)
        chosen.extend(sorted(front, key=lambda i: -dist[i])[: k - len(chosen)])
        break
    return chosen


def tournament(
    rank: npt.NDArray[np.int64],
    crowd: npt.NDArray[np.float64],
    rng: np.random.Generator,
    size: int = 2,
) -> int:
    """Crowded-comparison tournament: lower rank wins, ties broken by larger crowding."""
    cands = rng.choice(len(rank), size, replace=False)
    return int(min(cands, key=lambda i: (rank[i], -crowd[i])))
