"""Statistics for telling skill from luck.

* Probabilistic Sharpe Ratio (PSR): P(true SR > benchmark) given sample length, skew, kurtosis.
  Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier".
* Deflated Sharpe Ratio (DSR): PSR against the Sharpe the *best of N* noise strategies would
  reach by luck. Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
* Holm / Benjamini-Hochberg p-value adjustment for many simultaneous rule tests.
* Bootstrap confidence intervals.

Sharpe ratios here are **per period** (e.g. daily, not annualized) unless stated otherwise.
That is what the formulas expect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Moments:
    n: int
    mean: float
    std: float
    skew: float
    kurtosis: float  # non-excess (normal = 3)

    @property
    def sharpe(self) -> float:
        return self.mean / self.std if self.std > 0 else 0.0


def moments(x: Sequence[float] | FloatArray) -> Moments:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return Moments(n, float(a.mean()) if n else 0.0, 0.0, 0.0, 3.0)
    mu = float(a.mean())
    sd = float(a.std(ddof=1))
    if sd == 0:
        return Moments(n, mu, 0.0, 0.0, 3.0)
    z = (a - mu) / a.std(ddof=0)
    return Moments(n, mu, sd, float(np.mean(z**3)), float(np.mean(z**4)))


def probabilistic_sharpe(
    sr: float, n: int, skew: float = 0.0, kurtosis: float = 3.0, sr_benchmark: float = 0.0
) -> float:
    """P(true per-period SR > ``sr_benchmark``)."""
    if n < 2:
        return 0.5
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if denom <= 0:
        denom = 1e-12
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return _N.cdf(z)


def expected_max_sharpe(n_trials: int, sr_variance: float, sr_mean: float = 0.0) -> float:
    """Expected maximum per-period SR among ``n_trials`` *independent* skill-less strategies
    whose SR estimates have variance ``sr_variance``. The False Strategy Theorem."""
    if n_trials <= 1 or sr_variance <= 0:  # nothing to deflate
        return sr_mean
    n = float(n_trials)
    z = (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e))
    return sr_mean + math.sqrt(sr_variance) * z


def deflated_sharpe(
    returns: Sequence[float] | FloatArray,
    n_trials: int,
    sr_variance: float | None = None,
) -> float:
    """DSR: probability that the strategy's true SR beats the luck-adjusted benchmark.

    ``n_trials``: number of (effectively independent) strategies tried before picking this one.
    ``sr_variance``: variance of per-period SR estimates under the null. The default ~1/T (the
    sampling variance of a skill-less SR) is the textbook choice. Passing the empirical
    variance across a heterogeneous population is much stricter: genuine differences (cost
    drag, real edges) inflate it, and that rejected a planted, real edge in our tests.
    """
    m = moments(returns)
    if m.n < 2 or m.std == 0:
        return 0.0
    var = sr_variance if sr_variance is not None and sr_variance > 0 else 1.0 / m.n
    sr0 = expected_max_sharpe(n_trials, var)
    return probabilistic_sharpe(m.sharpe, m.n, m.skew, m.kurtosis, sr0)


def effective_trials(sharpes: Sequence[float] | FloatArray, correlation: float) -> int:
    """Rough count of independent trials: N_eff = N(1 - rho) + rho. GA populations are
    highly correlated (siblings share most of their genome), so raw N overstates the
    penalty. The full count is still tracked and used where we can't estimate rho."""
    n = len(sharpes)
    rho = min(max(correlation, 0.0), 1.0)
    return max(1, round(n * (1 - rho) + rho))


def t_test_mean(x: Sequence[float] | FloatArray) -> tuple[float, float]:
    """One-sided t-stat and (normal-approx) p-value for mean > 0."""
    m = moments(x)
    if m.n < 2 or m.std == 0:
        return 0.0, 1.0
    t = m.mean / (m.std / math.sqrt(m.n))
    return t, 1.0 - _N.cdf(t)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """BH adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, p[i] * m / rank)
        adj[i] = min(running, 1.0)
    return adj.tolist()


def bootstrap_ci(
    x: Sequence[float] | FloatArray,
    stat: Callable[[FloatArray], float] = lambda a: float(np.mean(a)),
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    block: int = 1,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI. ``block > 1`` resamples contiguous blocks (moving-block
    bootstrap) to respect autocorrelation in return series."""
    a = np.asarray(x, dtype=np.float64)
    n = len(a)
    if n < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = math.ceil(n / block)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([a[s : s + block] for s in starts])[:n]
        stats[b] = stat(sample)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
