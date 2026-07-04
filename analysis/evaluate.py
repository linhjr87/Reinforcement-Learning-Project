import numpy as np
from vasuki.env import Vasuki


def _act_as_b(opp_b, env):
    # opponent decides as if it were "A"; temporarily swap so it drives agent B
    env.agentA, env.agentB = env.agentB, env.agentA
    try:
        return opp_b.act(env)
    finally:
        env.agentA, env.agentB = env.agentB, env.agentA


def play_match(config, opp_a, opp_b, seed=0):
    env = Vasuki(**config)
    env.reset()
    done = False
    while not done:
        action_a = opp_a.act(env)
        action_b = _act_as_b(opp_b, env)
        _, _, done, _ = env.step_two(action_a, action_b)
    if env.agentA["score"] > env.agentB["score"]:
        return "A"
    if env.agentB["score"] > env.agentA["score"]:
        return "B"
    return "draw"


def round_robin(config, agents, n_games=100):
    matrix = {}
    for name_a, opp_a in agents.items():
        for name_b, opp_b in agents.items():
            if name_a == name_b:
                continue
            wins = sum(
                1 for g in range(n_games)
                if play_match(config, opp_a, opp_b, seed=g) == "A"
            )
            matrix[(name_a, name_b)] = wins / n_games
    return matrix
