import time
import torch
from stable_baselines3 import DQN
from vasuki.gym_wrapper import GymWrapper
from vasuki.opponents import RandomOpponent, PolicySnapshotOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def _pick_device(device):
    # "auto" -> prefer GPU if available; if "cuda" is requested without a GPU, warn and fall back to CPU
    if device in ("cuda", "auto") and not torch.cuda.is_available():
        if device == "cuda":
            print("[selfplay] WARNING: 'cuda' requested but no GPU found, falling back to CPU.")
        return "cpu"
    return device


def train_selfplay(total_timesteps=1_000_000, refresh_every=50_000,
                   save_path="models/dqn_selfplay.zip", seed=0, device="auto"):
    # Start against a random opponent; periodically freeze a snapshot of A (fictitious self-play)
    device = _pick_device(device)
    env = GymWrapper(CONFIG, RandomOpponent())
    model = DQN("MlpPolicy", env, verbose=1, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, batch_size=64, gamma=0.95,
                device=device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"[selfplay] Device: {model.device} (GPU: {gpu_name}) | "
          f"timesteps={total_timesteps} | refresh={refresh_every} | seed={seed}")

    t0 = time.time()
    steps_done = 0
    while steps_done < total_timesteps:
        chunk = min(refresh_every, total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        # Save then reload to create an INDEPENDENT frozen copy (not tied to the still-updating weights)
        model.save(save_path)
        snapshot = DQN.load(save_path)
        env.set_opponent(PolicySnapshotOpponent(snapshot))
        print(f"[selfplay] {steps_done}/{total_timesteps} steps | "
              f"refreshed opponent snapshot | {time.time() - t0:.1f}s")

    model.save(save_path)
    print(f"[selfplay] Done in {time.time() - t0:.1f}s. Saved model to {save_path}")
    return model
