"""Monte Carlo resampling of a trade sequence."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def _max_dd(paths: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    peak = np.maximum.accumulate(paths, axis=1)
    return np.asarray(np.max(1.0 - paths / peak, axis=1))


def monte_carlo(
    r: npt.ArrayLike,
    *,
    risk_frac: float = 0.005,
    n_paths: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Equity paths from trade R-multiples at fixed-fractional risk.

    * **shuffle**: same trades, random order. Shows how much of the historical drawdown was
      luck of sequencing.
    * **bootstrap**: trades resampled with replacement. Gives the distribution of final
      outcomes and P(loss).
    """
    arr = np.asarray(r, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return {"n_trades": len(arr)}
    rng = np.random.default_rng(seed)
    growth = 1.0 + risk_frac * arr
    shuffled = np.array([rng.permutation(growth) for _ in range(n_paths)])
    sh_paths = np.cumprod(np.hstack([np.ones((n_paths, 1)), shuffled]), axis=1)
    boot = growth[rng.integers(0, len(arr), (n_paths, len(arr)))]
    bt_paths = np.cumprod(np.hstack([np.ones((n_paths, 1)), boot]), axis=1)
    dd_sh = _max_dd(sh_paths)
    final = bt_paths[:, -1] - 1.0
    hist_path = np.cumprod(np.concatenate([[1.0], growth]))
    return {
        "n_trades": len(arr),
        "historical_max_dd": float(_max_dd(hist_path[None, :])[0]),
        "max_dd_p50": float(np.quantile(dd_sh, 0.5)),
        "max_dd_p95": float(np.quantile(dd_sh, 0.95)),
        "return_p05": float(np.quantile(final, 0.05)),
        "return_p50": float(np.quantile(final, 0.5)),
        "prob_loss": float(np.mean(final < 0)),
    }
