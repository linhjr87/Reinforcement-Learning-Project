import pickle
import numpy as np
from vasuki.env import Vasuki
from vasuki.features import get_input_states

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}
LR, DISCOUNT, FOOD_REWARD = 0.1, 0.95, 4


def _init_qtable(size):
    q = {}
    for i in range(-size + 1, size):
        for j in range(-size + 1, size):
            for k in range(-size + 1, size):
                for l in range(-size + 1, size):
                    for m in range(4):
                        q[((i, j), (k, l), m)] = [np.random.uniform(-5, 0) for _ in range(3)]
    return q


def train_tabular(num_episodes=50_000, save_path="models/best_qtable.pkl", seed=0):
    np.random.seed(seed)
    env = Vasuki(**CONFIG)
    q_table = _init_qtable(CONFIG["n"])
    epsilon, eps_decay = 0.9, 0.9998
    for ep in range(num_episodes):
        env.reset()
        for _ in range(CONFIG["game_length"]):
            obs = get_input_states(env, "A")
            if np.random.random() > epsilon:
                action_a = int(np.argmax(q_table[obs]))
            else:
                action_a = int(env.action_space.sample())
            action_b = int(env.action_space.sample())
            rewardA, _, done, _ = env.step_two(action_a, action_b)
            new_obs = get_input_states(env, "A")
            max_future_q = np.max(q_table[new_obs])
            current_q = q_table[obs][action_a]
            if rewardA >= FOOD_REWARD:
                new_q = FOOD_REWARD
            else:
                new_q = (1 - LR) * current_q + LR * (rewardA + DISCOUNT * max_future_q)
            q_table[obs][action_a] = new_q
            if done:
                break
        epsilon *= eps_decay
    with open(save_path, "wb") as f:
        pickle.dump(q_table, f)
    return q_table
