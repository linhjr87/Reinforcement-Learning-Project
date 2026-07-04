import pickle
from vasuki.env import Vasuki
from vasuki.opponents import RandomOpponent, TabularOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def test_random_opponent_in_range():
    env = Vasuki(**CONFIG)
    opp = RandomOpponent()
    for _ in range(20):
        assert opp.act(env) in (0, 1, 2)


def test_tabular_opponent_uses_qtable():
    env = Vasuki(**CONFIG)
    with open("models/qtable_new.pkl", "rb") as f:
        q_table = pickle.load(f)
    opp = TabularOpponent(q_table)
    assert opp.act(env) in (0, 1, 2)
