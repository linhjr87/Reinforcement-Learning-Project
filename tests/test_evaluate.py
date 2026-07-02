from vasuki.opponents import RandomOpponent
from analysis.evaluate import play_match, round_robin

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def test_play_match_returns_result():
    r = play_match(CONFIG, RandomOpponent(), RandomOpponent(), seed=0)
    assert r in ("A", "B", "draw")


def test_round_robin_shape():
    agents = {"rand1": RandomOpponent(), "rand2": RandomOpponent()}
    matrix = round_robin(CONFIG, agents, n_games=5)
    assert ("rand1", "rand2") in matrix
    assert 0.0 <= matrix[("rand1", "rand2")] <= 1.0
