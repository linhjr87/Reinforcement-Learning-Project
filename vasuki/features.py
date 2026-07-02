import numpy as np


def get_input_states(env, player="A"):
    playerA, playerB = env.agentA, env.agentB
    if player != "A":
        playerA, playerB = playerB, playerA
    foods = env.live_foodspawn_space
    head = playerA["head"]
    enemy_coord = (
        int(playerA["state"][0] - playerB["state"][0]),
        int(playerA["state"][1] - playerB["state"][1]),
    )
    nearest_food, distance = None, np.inf
    for food in foods:
        d = np.linalg.norm(playerA["state"] - food)
        if d < distance:
            distance, nearest_food = d, food
    food_coord = (
        int(playerA["state"][0] - nearest_food[0]),
        int(playerA["state"][1] - nearest_food[1]),
    )
    return food_coord, enemy_coord, head


def to_vector(env, player="A"):
    food, enemy, head = get_input_states(env, player)
    n = env.n
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = np.clip(food[0] / n, -1, 1)
    vec[1] = np.clip(food[1] / n, -1, 1)
    vec[2] = np.clip(enemy[0] / n, -1, 1)
    vec[3] = np.clip(enemy[1] / n, -1, 1)
    vec[4 + head] = 1.0
    return vec
