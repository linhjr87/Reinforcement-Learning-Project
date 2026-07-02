import argparse
from stable_baselines3 import DQN
from vasuki.gym_wrapper import GymWrapper
from vasuki.opponents import RandomOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def train_dqn(total_timesteps=200_000, save_path="models/dqn_random.zip", seed=0):
    env = GymWrapper(CONFIG, RandomOpponent())
    model = DQN("MlpPolicy", env, verbose=0, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, batch_size=64, gamma=0.95)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--out", type=str, default="models/dqn_random.zip")
    args = p.parse_args()
    train_dqn(args.timesteps, args.out)
    print(f"Saved model to {args.out}")
