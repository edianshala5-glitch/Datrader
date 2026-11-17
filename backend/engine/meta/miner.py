"""Mine candidate context rules from a strategy's trades (mining period only).

1. **What matters**: a gradient-boosted tree model predicts trade R from the context features.
   Permutation importance ranks them (shown in the UI as "drivers of this strategy").
2. **Rules from trees**: shallow regression trees with large leaves. Every leaf whose trades
   beat the baseline becomes a rule (its root-to-leaf path).
3. **Beam search**: greedy search over interpretable clauses (weekday sets, contiguous NY
   hour windows, session flags, regime states, quantile thresholds), up to 3 clauses.

Candidates must beat the baseline (positive lift) and are ranked by the t-statistic of the
kept trades' mean R: rules whose trades make money significantly, not merely less badly.
Only the top few go on to out-of-sample testing, because every OOS test costs statistical
power in the multiple-testing correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor

from engine.meta.rules import FEATURES, Clause, Rule


@dataclass(frozen=True)
class MinerConfig:
    min_coverage: float = (
        0.01  # narrow contexts (e.g. one weekday x two hours ~ 2%) must be findable
    )
    min_trades: int = 30
    max_clauses: int = 3
    beam_width: int = 8
    tree_depths: tuple[int, ...] = (2, 3)
    top_k: int = 20  # candidates forwarded to OOS testing
    seed: int = 0


@dataclass
class Candidate:
    rule: Rule
    n: int
    coverage: float
    mean_r: float
    win_rate: float
    profit_factor: float | None
    lift: float
    score: float  # t-stat of the kept trades' mean R


def _stats(
    r: np.ndarray, mask: np.ndarray, base_mean: float
) -> tuple[int, float, float, float | None, float, float]:
    kept = r[mask]
    n = len(kept)
    if n < 2:
        return n, 0.0, 0.0, None, 0.0, -math.inf
    mean = float(kept.mean())
    sd = float(kept.std(ddof=1)) or 1e-9
    gains, losses = kept[kept > 0].sum(), -kept[kept <= 0].sum()
    pf = float(gains / losses) if losses > 0 else None
    lift = mean - base_mean
    # Rank by how significantly the *kept* trades make money (t-stat of mean R). Ranking by
    # lift significance favoured huge-coverage rules with tiny, break-even gains.
    return n, mean, float((kept > 0).mean()), pf, lift, mean / (sd / math.sqrt(n))


def candidate_clauses(x: pd.DataFrame) -> list[Clause]:
    out: list[Clause] = []
    for name, f in FEATURES.items():
        if name not in x:
            continue
        col = x[name]
        if name == "weekday":
            for a in range(5):
                for b in range(a, 5):
                    if b - a < 4:
                        out.append(Clause(name, "in", tuple(range(a, b + 1))))
        elif name == "hour_ny":
            for a in range(24):
                for length in (1, 2, 3, 4, 6):
                    if a + length <= 24:
                        out.append(Clause(name, "in", tuple(range(a, a + length))))
        elif f.kind == "bool":
            if col.any() and (~col).any():
                out += [Clause(name, "in", (True,)), Clause(name, "in", (False,))]
        elif f.kind == "ordinal":
            vals = sorted(int(v) for v in col.unique())
            for v in vals:
                out.append(Clause(name, "in", (v,)))
            for i in range(len(vals) - 1):
                out.append(Clause(name, "in", tuple(vals[: i + 1])))
                out.append(Clause(name, "in", tuple(vals[i + 1 :])))
        else:
            finite = col[col < 1e4] if name.startswith("bars_since") else col
            if finite.nunique() < 3:
                continue
            qs = sorted(
                {
                    float(np.round(q, 4))
                    for q in np.quantile(finite, [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
                }
            )
            for q in qs:
                if name.startswith("bars_since"):
                    q = float(int(q))
                out += [Clause(name, "<=", q), Clause(name, ">", q)]
    return out


def importance(x: pd.DataFrame, r: np.ndarray, seed: int = 0) -> list[dict[str, Any]]:
    if len(x) < 50:
        return []
    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        min_samples_leaf=max(20, len(x) // 50),
        random_state=seed,
    )
    xf = x.astype(float)
    model.fit(xf, r)
    pi = permutation_importance(model, xf, r, n_repeats=5, random_state=seed)
    order = np.argsort(-pi.importances_mean)
    return [
        {
            "feature": x.columns[i],
            "label": FEATURES[x.columns[i]].label,
            "importance": float(pi.importances_mean[i]),
            "std": float(pi.importances_std[i]),
        }
        for i in order
    ]


def _tree_rules(x: pd.DataFrame, r: np.ndarray, depth: int, min_leaf: int, seed: int) -> list[Rule]:
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
    xf = x.astype(float)
    tree.fit(xf, r)
    t = tree.tree_
    rules: list[Rule] = []

    def walk_node(node: int, path: list[tuple[str, str, float]]) -> None:
        if t.children_left[node] == -1:
            if path:
                rules.append(
                    Rule(tuple(_to_clause(x, f, op, th) for f, op, th in path), f"tree_d{depth}")
                )
            return
        name, th = x.columns[t.feature[node]], float(t.threshold[node])
        walk_node(t.children_left[node], [*path, (name, "<=", th)])
        walk_node(t.children_right[node], [*path, (name, ">", th)])

    walk_node(0, [])
    return rules


def _to_clause(x: pd.DataFrame, name: str, op: str, th: float) -> Clause:
    f = FEATURES[name]
    if f.kind in ("ordinal", "bool"):
        vals = sorted(v for v in x[name].unique())
        keep = [v for v in vals if (float(v) <= th) == (op == "<=")]
        if f.kind == "bool":
            keep = [bool(v) for v in keep]
        return Clause(name, "in", tuple(int(v) if f.kind == "ordinal" else v for v in keep))
    if name.startswith("bars_since"):
        th = float(math.floor(th))
    return Clause(name, op, round(th, 4))


def _merge(rule: Rule) -> Rule:
    """Collapse repeated clauses on one feature (trees split the same feature twice)."""
    by: dict[str, list[Clause]] = {}
    for c in rule.clauses:
        by.setdefault(c.feature, []).append(c)
    out = []
    for name, cs in by.items():
        if len(cs) == 1:
            out.append(cs[0])
        elif all(c.op == "in" for c in cs):
            common = set(cs[0].value)
            for c in cs[1:]:
                common &= set(c.value)
            out.append(Clause(name, "in", tuple(sorted(common))))
        else:
            le = [c.value for c in cs if c.op == "<="]
            gt = [c.value for c in cs if c.op == ">"]
            if le:
                out.append(Clause(name, "<=", min(le)))
            if gt:
                out.append(Clause(name, ">", max(gt)))
    return Rule(tuple(out), rule.miner)


def _diverse(cands: list[Candidate], width: int) -> list[Candidate]:
    """Best candidate per *set of features*, then the top ``width``. Without this, near-identical
    thresholds on one feature (ATR > q1, ATR > q2, ...) crowd the beam and the search never
    combines e.g. a weekday with an hour window."""
    best: dict[frozenset[str], Candidate] = {}
    for c in cands:
        sig = frozenset(cl.feature for cl in c.rule.clauses)
        if sig not in best or c.score > best[sig].score:
            best[sig] = c
    return sorted(best.values(), key=lambda c: -c.score)[:width]


def mine(x: pd.DataFrame, r: np.ndarray, cfg: MinerConfig | None = None) -> list[Candidate]:
    cfg = cfg or MinerConfig()
    n_all = len(r)
    base = float(r.mean()) if n_all else 0.0
    min_n = max(cfg.min_trades, int(cfg.min_coverage * n_all))
    seen: dict[tuple[Any, ...], Candidate] = {}

    def consider(rule: Rule) -> Candidate | None:
        rule = _merge(rule)
        if any(c.op == "in" and len(c.value) == 0 for c in rule.clauses):
            return None
        key = rule.key()
        if key in seen:
            return seen[key]
        mask = rule.mask(x)
        n, mean, wr, pf, lift, score = _stats(r, mask, base)
        if n < min_n or n == n_all:
            return None
        cand = Candidate(rule, n, n / n_all, mean, wr, pf, lift, score)
        seen[key] = cand
        return cand

    for depth in cfg.tree_depths:
        for rule in _tree_rules(x, r, depth, min_n, cfg.seed):
            consider(rule)

    clauses = candidate_clauses(x)
    beam: list[Candidate] = []
    for c in clauses:
        cand = consider(Rule((c,), "beam"))
        if cand:
            beam.append(cand)
    beam = _diverse(beam, cfg.beam_width)
    for _ in range(cfg.max_clauses - 1):
        nxt: list[Candidate] = []
        for b in beam:
            used = {c.feature for c in b.rule.clauses}
            for c in clauses:
                if c.feature in used:
                    continue
                cand = consider(Rule((*b.rule.clauses, c), "beam"))
                if cand and cand.score > b.score:
                    nxt.append(cand)
        if not nxt:
            break
        beam = _diverse(nxt, cfg.beam_width)

    ranked = sorted(
        (c for c in seen.values() if c.lift > 0 and c.mean_r > 0), key=lambda c: -c.score
    )
    return ranked[: cfg.top_k]
