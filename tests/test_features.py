import numpy as np
from vasuki.env import Vasuki
from vasuki.features import get_input_states, to_vector

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def test_get_input_states_shape():
    env = Vasuki(**CONFIG)
    food, enemy, head = get_input_states(env, "A")
    assert len(food) == 2 and len(enemy) == 2
    assert head in (0, 1, 2, 3)


def test_to_vector_normalized():
    env = Vasuki(**CONFIG)
    vec = to_vector(env, "A")
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    assert np.all(vec >= -1.0) and np.all(vec <= 1.0)
    assert vec[4:].sum() == 1.0  # head one-hot
