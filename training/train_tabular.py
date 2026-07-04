import pickle
import time
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


def train_tabular(num_episodes=50_000, save_path="models/qtable_new.pkl", seed=0,
                  log_every=1_000):
    np.random.seed(seed)
    env = Vasuki(**CONFIG)
    # Q-learning dạng bảng chạy thuần NumPy trên CPU (không dùng GPU).
    print(f"[train_tabular] Thiết bị: CPU (thuần NumPy) | episodes={num_episodes} | seed={seed}")
    q_table = _init_qtable(CONFIG["n"])
    epsilon, eps_decay = 0.9, 0.9998
    t0 = time.time()
    ep_rewards = []
    for ep in range(num_episodes):
        env.reset()
        ep_reward = 0.0
        for _ in range(CONFIG["game_length"]):
            obs = get_input_states(env, "A")
            if np.random.random() > epsilon:
                action_a = int(np.argmax(q_table[obs]))
            else:
                action_a = int(env.action_space.sample())
            action_b = int(env.action_space.sample())
            rewardA, _, done, _ = env.step_two(action_a, action_b)
            ep_reward += rewardA
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
        ep_rewards.append(ep_reward)
        if log_every and (ep + 1) % log_every == 0:
            avg_reward = float(np.mean(ep_rewards[-log_every:]))
            print(f"[train_tabular] ep {ep + 1}/{num_episodes} | "
                  f"epsilon={epsilon:.3f} | reward_tb={avg_reward:.2f} | "
                  f"{time.time() - t0:.1f}s")
    with open(save_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"[train_tabular] Hoàn tất sau {time.time() - t0:.1f}s. Đã lưu Q-table vào {save_path}")
    return q_table
