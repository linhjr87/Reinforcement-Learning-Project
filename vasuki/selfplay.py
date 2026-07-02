from stable_baselines3 import DQN
from vasuki.gym_wrapper import GymWrapper
from vasuki.opponents import RandomOpponent, PolicySnapshotOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def train_selfplay(total_timesteps=1_000_000, refresh_every=50_000,
                   save_path="models/dqn_selfplay.zip", seed=0):
    # Bắt đầu với đối thủ random; đóng băng snapshot A theo chu kỳ (fictitious self-play)
    env = GymWrapper(CONFIG, RandomOpponent())
    model = DQN("MlpPolicy", env, verbose=0, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, batch_size=64, gamma=0.95)

    steps_done = 0
    while steps_done < total_timesteps:
        chunk = min(refresh_every, total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        # Lưu rồi load lại để tạo bản đóng băng ĐỘC LẬP (không dính weights đang cập nhật)
        model.save(save_path)
        snapshot = DQN.load(save_path)
        env.set_opponent(PolicySnapshotOpponent(snapshot))

    model.save(save_path)
    return model
