# Vasuki: Multi-Agent Snake with Tabular Q-Learning and DQN

Vasuki is a two-agent, grid-world snake environment (two snakes racing for food on an `n x n`
grid) used to compare a rule-based bot, tabular Q-learning, and Deep Q-Networks (DQN) — including
a fictitious self-play variant — as opponents in head-to-head matches.

## Problem

The original project pits a Bot A against a hard-coded, rule-based random Bot B and trains Bot A
with tabular Q-learning. This works on a small grid, but has two core issues:

1. **A random opponent is a weak adversary.** Bot B never adapts, so Bot A only ever learns to
   beat a "dumb" opponent (the original arena reports lopsided results like A=100 / B=0), which
   does not necessarily generalize to stronger or better-adapted opponents.
2. **Tabular Q-learning does not scale.** The state used for tabular learning is
   `(food_coord, enemy_coord, head)`, where each coordinate component ranges over `2n - 1` values
   for an `n x n` grid. The Q-table therefore has on the order of `(2n - 1)^4 * 4` entries —
   roughly 200K entries at `n = 8`, and over 1.1M at `n = 12`. The table grows as `O(n^4)`, making
   tabular learning infeasible on larger grids and unable to generalize across states it hasn't
   visited.

## Contributions

This project migrates and extends the original codebase with:

- **Gymnasium migration** — a `GymWrapper` (`vasuki/gym_wrapper.py`) wraps the core `Vasuki`
  environment in the modern `gymnasium.Env` API (5-tuple `step` returning
  `obs, reward, terminated, truncated, info`), enabling use of standard RL libraries, while the
  underlying two-agent env exposes a pure `step_two()` for direct control of both snakes.
- **DQN** — a Deep Q-Network (Stable-Baselines3) agent (`training/train_dqn.py`) trained on a
  fixed-size 8-dimensional feature vector (`vasuki/features.py::to_vector`:
  `[food_dx, food_dy, enemy_dx, enemy_dy, head_onehot(4)]`), whose parameter count is independent
  of grid size `n`, unlike the tabular approach.
- **Fictitious self-play** — `training/train_selfplay.py` / `vasuki/selfplay.py` periodically
  freeze a snapshot of the current DQN policy and switch the opponent to that frozen snapshot
  (`PolicySnapshotOpponent`), so the agent trains against progressively stronger versions of
  itself instead of a static random bot.

## Structure

```
vasuki/        # env.py, features.py, opponents.py, gym_wrapper.py, selfplay.py
training/      # train_tabular.py, train_dqn.py, train_selfplay.py
analysis/      # evaluate.py (round-robin win-rate), experiments.ipynb
demo/          # demo.ipynb (pick 2 policies -> export a match video)
models/        # qtable_new.pkl, dqn_random.zip, dqn_selfplay.zip
tests/         # env, features, opponents, wrapper, and smoke tests
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt

# Tabular Q-learning baseline -> models/qtable_new.pkl (slow; reduce num_episodes for a quick run)
python -m training.train_tabular

# DQN vs. a random opponent -> models/dqn_random.zip
python -m training.train_dqn --timesteps 200000 --out models/dqn_random.zip

# DQN via fictitious self-play -> models/dqn_selfplay.zip
python -m training.train_selfplay --timesteps 1000000 --refresh 50000 --out models/dqn_selfplay.zip

# Tests
pytest tests/ -q
```

### GPU and training logs

The DQN trainers (`train_dqn`, `train_selfplay`) accept a `--device` flag: `auto` (default; uses
the GPU if one is available, otherwise CPU), `cuda`, or `cpu`. On startup each prints the resolved
device and GPU name, then streams progress:

```bash
python -m training.train_dqn --timesteps 200000 --device auto --log-interval 10
python -m training.train_selfplay --timesteps 1000000 --refresh 50000 --device cuda
```

- `train_dqn` / `train_selfplay` run with Stable-Baselines3 `verbose=1`, so SB3 prints its
  loss/reward/fps table (`--log-interval` controls how often, in episodes); self-play also logs
  each snapshot refresh with elapsed time.
- `train_tabular` is pure NumPy and always runs on **CPU** (no GPU). Call it from Python with
  `train_tabular(..., log_every=1000)` to print episode progress, current `epsilon`, and the
  rolling average reward.

> Note: for the small `MlpPolicy` networks here, DQN can run faster on CPU than GPU because
> per-step kernel-launch overhead dominates the tiny batches. `--device auto` uses the GPU when
> present; pass `--device cpu` if you find it quicker.

Once the models above exist, open `analysis/experiments.ipynb` to plot learning curves and the
win-rate matrix, and `demo/demo.ipynb` to export a video of a match between two chosen policies.

## Results (fill in after training)

Win rate of the row agent (A) against the column agent (B), evaluated over `n_games` games per
pair via `analysis.evaluate.round_robin`. **This table is a placeholder — fill in with real
numbers after training all agents and running `analysis/experiments.ipynb`.**

| A \ B        | Random | Tabular | DQN-random | DQN-selfplay |
|--------------|:------:|:-------:|:----------:|:------------:|
| Random       |   –    |   TBD   |    TBD     |     TBD      |
| Tabular      |  TBD   |    –    |    TBD     |     TBD      |
| DQN-random   |  TBD   |   TBD   |     –      |     TBD      |
| DQN-selfplay |  TBD   |   TBD   |    TBD     |      –       |

## Future directions

- Multi-agent: train both agents simultaneously and independently, observing "arms race" dynamics.
- League-lite: maintain a pool of past snapshots and sample opponents from it at random.
- Stronger algorithms (PPO, AlphaZero-style self-play with MCTS).

## Attribution

This project is adapted and migrated from
[`aswintechguy/Reinforcement-Learning-Projects`](https://github.com/aswintechguy/Reinforcement-Learning-Projects),
which provided the original Vasuki snake environment and the rule-based/tabular Q-learning
baseline. This repository extends that Snake Game into Deep RL with DQN and fictitious self-play,
for learning purposes.
