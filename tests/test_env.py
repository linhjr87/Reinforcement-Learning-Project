import numpy as np
from vasuki.env import Vasuki

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def test_env_constructs():
    env = Vasuki(**CONFIG)
    assert env.n == 8
    assert env.action_space.n == 3
    assert env.agentA["state"].shape == (2,)
    assert len(env.live_foodspawn_space) == 4
