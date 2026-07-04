# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Vasuki is a two-agent grid-world snake environment (two snakes racing for food on an `n x n`
grid). The project compares four opponents head-to-head: a rule-based random bot, tabular
Q-learning, DQN (Stable-Baselines3), and a DQN trained via fictitious self-play. The point of the
project is the comparison, so most work touches the *interfaces between* the env, the observation
encoders, and the opponents rather than any single file.

## Commands

Run everything from the repo root. There is no `setup.py`/`pyproject.toml`; imports are absolute
(`from vasuki.env import ...`), so the package is only importable when the working directory is the
repo root. Always invoke training as `python -m training.<mod>`, never as a direct script path.

```bash
pip install -r requirements.txt      # needs torch + stable_baselines3 + opencv

# Tests (fast — smoke tests train tiny models, so torch/SB3 must be installed)
pytest tests/ -q
pytest tests/test_env.py -q                       # single file
pytest tests/test_env.py::test_env_constructs -q  # single test

# DQN vs. a random opponent -> models/dqn_random.zip
python -m training.train_dqn --timesteps 200000 --out models/dqn_random.zip --device auto

# DQN via fictitious self-play -> models/dqn_selfplay.zip
python -m training.train_selfplay --timesteps 1000000 --refresh 50000 --out models/dqn_selfplay.zip --device auto
```

Note: all three trainers now have argparse CLI entry points. `train_tabular` defaults to 50k
episodes and is slow, so also usable from Python/a notebook (`training/train_tabular.ipynb`):

```bash
# Tabular Q-learning -> models/qtable_new.pkl (WARNING: overwrites the committed table)
python -m training.train_tabular --episodes 50000 --out models/qtable_new.pkl
```

`models/qtable_new.pkl` is already committed, and `tests/test_opponents.py` reads it from disk, so
don't delete it (and beware `--out models/qtable_new.pkl` overwrites it).

### Device selection and training logs

Both DQN trainers take a `device` arg (CLI `--device`, values `auto`/`cuda`/`cpu`, default `auto`)
threaded into the `DQN(...)` constructor; a shared `_pick_device()` helper (copy-pasted in
`train_dqn.py` and `selfplay.py`) falls back to CPU with a warning if `cuda` is requested without a
GPU. Both run SB3 with `verbose=1` and print a startup line with the resolved `model.device` + GPU
name; self-play also logs each snapshot refresh. `train_tabular` is pure NumPy / **CPU-only** (no
`device` arg) and takes a `log_every` kwarg for periodic episode/epsilon/reward logging. Note: for
these small `MlpPolicy` nets, CPU is often faster than GPU (kernel-launch overhead on tiny batches).

## Architecture

### Two-layer environment (the key structural fact)

There are **two** env objects with **different `step` contracts**, and confusing them is the most
common mistake:

- `vasuki/env.py::Vasuki` — the core two-agent env. Its method is `step_two(actionA, actionB)` and
  it returns a **4-tuple** `(rewardA, rewardB, done, info)` — *not* the Gymnasium API. It drives
  both snakes at once and is what `analysis/evaluate.py` and `demo/demo.ipynb` use directly.
- `vasuki/gym_wrapper.py::GymWrapper` — wraps `Vasuki` into a single-agent `gymnasium.Env`. The
  learning agent always controls snake A; an injected `opponent` controls snake B. Its `step(action)`
  returns the standard **5-tuple** `(obs, reward, terminated, truncated, info)` and is what SB3 sees.

### Opponents and the A/B perspective swap (critical, appears in 3 places)

Every opponent implements a single method `act(env) -> int` and is **always written from snake A's
point of view** (`vasuki/opponents.py`: `RandomOpponent`, `TabularOpponent`, `PolicySnapshotOpponent`).
To make an opponent actually drive snake B, callers temporarily swap `env.agentA` and `env.agentB`
around the `act()` call, then swap back. This exact pattern is duplicated in three places and must
stay consistent:

- `GymWrapper._opponent_action()`
- `analysis/evaluate.py::_act_as_b()`
- `demo/demo.ipynb`'s `Runner._act_as_b`

Any new opponent you add must assume it is snake A; the swap wiring gives it B when needed.

### Two observation encoders (`vasuki/features.py`)

Both take a `player="A"|"B"` argument that internally swaps perspective:

- `get_input_states(env, player)` -> a **hashable tuple** `(food_coord, enemy_coord, head)`, used as
  tabular Q-table keys. `food_coord`/`enemy_coord` are integer deltas in `[-(n-1), n-1]`.
- `to_vector(env, player)` -> a fixed **8-dim float32 vector**
  `[food_dx/n, food_dy/n, enemy_dx/n, enemy_dy/n, head_onehot(4)]`, clipped to `[-1, 1]`. This is the
  DQN observation and its size is independent of `n` (the whole motivation for DQN over tabular).

### Fictitious self-play (`vasuki/selfplay.py`)

`train_selfplay` trains one DQN and, every `refresh_every` steps, **saves the model to disk and
reloads it** to get a weight-independent frozen snapshot, then installs it as the opponent via
`env.set_opponent(PolicySnapshotOpponent(snapshot))`. The save/reload (rather than reusing the live
model object) is deliberate — it prevents the frozen opponent from tracking the still-updating
weights. Training continues across chunks with `reset_num_timesteps=False`.

### Shared config and env invariants

- `CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}` is
  **copy-pasted** across `train_tabular.py`, `train_dqn.py`, `selfplay.py`, the demo, and every test.
  If you change env parameters, change all copies.
- Food spawn locations come from the N-Queens solution (`nqueens.Queen`) to spread them out; only
  `n//2` are live at any time. Several places **hardcode `assert len(...) == 4`**, which is only
  correct for `n = 8` (`8//2 == 4`). Changing `n` will trip these assertions until they're generalized.
- Collision/tie handling in `step_two` uses its own hardcoded reward formulas (`5*abs(...)`,
  `-3*abs(...)`) and respawns the losing snake — this is separate from the `rewards` dict, which only
  covers Food/Movement/Illegal.
- `Vasuki.__init__` loads `assets/agentA.png`, `assets/agentB.png`, `assets/prey.png` via OpenCV, so
  those files must exist to construct the env (they are committed).
- `Vasuki.render()` reads `self.history[-2]` (the previous step), so it is only valid to call once
  `len(env.history) >= 2`, i.e. from the second `step_two` onward. The demo notebook relies on this.

## Conventions

- All comments, docstrings, and log messages are in English; keep new code English too.
- Trained `.zip` models and rendered videos (`*.avi`, `*.mp4`) are gitignored; only
  `models/qtable_new.pkl` is versioned.
- `analysis/experiments.ipynb` (learning curves + win-rate matrix) and `demo/demo.ipynb` (export a
  match video) consume the trained artifacts above and expect them to already exist on disk.
