import numpy as np
from vasuki.env import Vasuki

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def test_env_constructs():
    env = Vasuki(**CONFIG)
    assert env.n == 8
    assert env.action_space.n == 3
    assert env.agentA["state"].shape == (2,)
    assert len(env.live_foodspawn_space) == 4


def test_step_two_returns_four_values():
    env = Vasuki(**CONFIG)
    out = env.step_two(1, 1)
    assert len(out) == 4
    rewardA, rewardB, done, info = out
    assert isinstance(done, bool)
    assert "agentA" in info and "agentB" in info


def test_tie_collision_does_not_crash():
    env = Vasuki(**CONFIG)
    env.agentA["score"] = 10
    env.agentB["score"] = 10
    env.agentA["state"] = np.array([3, 3])
    env.agentB["state"] = np.array([3, 3])
    env.agentA["head"] = 1
    env.agentB["head"] = 1
    rewardA, rewardB, done, info = env.step_two(1, 1)  # phải chạy không lỗi
    assert (env.agentA["state"] != env.agentB["state"]).any()
