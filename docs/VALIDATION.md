# Validation: how the system tells edge from luck

A swarm that tests thousands of strategies will *always* find some with great backtests.
Every strategy therefore goes through this harness before it can be called `validated`,
and every number on the leaderboard comes from out-of-sample data.

## Data splits

```
|<----------------------- research (80%) ----------------------->|<- holdout (20%) ->|
| train 1 |e| test 1 |
|      train 2      |e| test 2 |                (anchored walk-forward, 5 folds)
                                                                  ^ evaluated ONCE
```

- **Walk-forward folds:** each strategy is measured on test windows it wasn't selected on.
  The embargo `e` (1 day) stops trades or indicator state from leaking across the boundary.
- **Locked holdout:** the final 20% is never seen during research. It is evaluated once per
  strategy, at promotion time. The database enforces this with a partial unique index: a
  second holdout row for the same strategy is rejected even if application code is bypassed.

## Gates (all must pass for `validated`)

| Gate | Default | Why |
|---|---|---|
| OOS trades | ≥ 100 | Too few trades = no statistical power |
| OOS expectancy | > 0R after costs | Must make money where it wasn't fitted |
| Deflated Sharpe | ≥ 0.95 | P(true Sharpe > what the best of *N* random strategies would show by luck) |
| Positive folds | ≥ 60% | Edge must be consistent across time, not one lucky period |
| Parameter robustness | ≥ 60% of ±15% neighbours profitable | Real edges don't vanish when a parameter moves slightly |
| Cost stress | > 0R at 2× commission and slippage, 1.5× spread | Survives worse execution |
| PBO (when a cohort is available) | ≤ 0.3 | 0.5 = picking the best backtest is no better than random |

**Overfit risk** (0 robust … 1 likely noise) is a weighted blend of the same components:
DSR 30%, fold consistency 15%, walk-forward efficiency 15%, robustness 15%, sample size 10%,
PBO 10%, costs 5%. The leaderboard filters on it.

## The trial count matters most

The Deflated Sharpe needs `n_trials`, the number of strategies tried before picking this
one. The swarm records it on every run (`evolution_runs.evaluations`).
Experiment (`tests/unit/test_validation.py::test_lucky_winner_needs_trial_correction`):

> Best of 50 random strategies on **edge-free, cost-free** synthetic data: in-sample
> Sharpe 1.09, positive folds, robust to parameter changes.
> Validated as if it were the only strategy tried: **overfit risk 0.11**, DSR 0.90.
> Validated honestly (n_trials = 50): **DSR 0.17, rejected**. PBO of the selection: 0.50.

## Which trials to count (swarm runs)

Selection bias exists only when a strategy is chosen and measured **on the same data**.
The pipeline keeps the layers apart:

```
GA training window  →  walk-forward OOS windows  →  locked holdout
(the swarm selects)    (validation measures)        (promotion, once)
```

The swarm never sees the OOS windows, so it is not charged for every genome it tried
(1,500+ in a small run). That would reject real edges it found honestly: a near-exact
rediscovery of a planted edge failed at DSR 0.56 that way. Instead the Deflated Sharpe uses:

`n_trials = candidates validated now + every strategy previously judged on the same
instrument/timeframe OOS data`

The second term is the OOS-reuse counter. Repeatedly validating on the same windows
gradually raises the bar, as it should. The raw search count is still recorded on every
report (`search_evaluations`) for transparency. When selection *does* use the measured data,
as in the lucky-winner experiment below, every trial counts.

## Swarm end-to-end controls

4 islands x 30 strategies x 12 generations (~1,550 evaluations), top 10 validated:

| Data | Runs | Strategies validated |
|---|---|---|
| Edge-free | 4 | **0 / 40** |
| Planted 08:00-11:00 NY drift | 4 | 3 / 40 (all from one run, all exploiting the planted window) |

At this deliberately small budget, discovery of a narrow time-window edge is hit-or-miss.
Certification never certified noise. Finding *when* a strategy works is the Meta-Analyzer's
job (Step 7). Production runs use far larger budgets.

## Calibration: planted-edge experiments

The synthetic generator can plant a known edge (an upward drift 08:00-11:00 NY). The rule
"buy the 08:15 NY open, hold 2h" was validated at different edge strengths, with real costs:

| Planted edge | OOS expectancy | DSR | Overfit risk | Verdict |
|---|---|---|---|---|
| none | -0.03R | 0.16 | 0.78 | rejected (5 gates) |
| tiny | +0.01R | 0.65 | 0.34 | rejected (DSR only: real but too weak to prove) |
| medium | +0.05R | 0.95 | 0.13 | **validated** |
| strong | +0.09R | 1.00 | 0.05 | **validated** |

## Reading PBO correctly

PBO measures whether in-sample *rankings* persist, not whether anything makes money.
On edge-free data with costs, low-frequency strategies are consistently *less bad*, so
PBO can be low (0.12) while the selected strategy still loses out of sample 84% of the time.
That's why PBO is one input among several and never a verdict on its own.

## Commands

```bash
uv run swarm validate --template sweep_reversal --start 2019-01-01 --end 2025-01-01 --trials 500
curl -X POST localhost:8000/api/v1/validate -d '{"genome": {...}, "start": "...", "end": "...", "n_trials": 500}'
```
