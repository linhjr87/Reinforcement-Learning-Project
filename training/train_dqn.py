import argparse
import time
import torch
from stable_baselines3 import DQN
from vasuki.gym_wrapper import GymWrapper
from vasuki.opponents import RandomOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def _pick_device(device):
    # "auto" -> ưu tiên GPU nếu có; nếu người dùng chỉ định "cuda" mà không có GPU thì cảnh báo
    if device in ("cuda", "auto") and not torch.cuda.is_available():
        if device == "cuda":
            print("[train_dqn] CẢNH BÁO: yêu cầu 'cuda' nhưng không thấy GPU, chuyển sang CPU.")
        return "cpu"
    return device


def train_dqn(total_timesteps=200_000, save_path="models/dqn_random.zip", seed=0,
              device="auto", log_interval=10):
    device = _pick_device(device)
    env = GymWrapper(CONFIG, RandomOpponent())
    model = DQN("MlpPolicy", env, verbose=1, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, batch_size=64, gamma=0.95,
                device=device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"[train_dqn] Thiết bị: {model.device} (GPU: {gpu_name}) | "
          f"timesteps={total_timesteps} | seed={seed}")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    model.save(save_path)
    print(f"[train_dqn] Hoàn tất sau {time.time() - t0:.1f}s. Đã lưu model vào {save_path}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--out", type=str, default="models/dqn_random.zip")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cpu (auto sẽ dùng GPU nếu có)")
    p.add_argument("--log-interval", type=int, default=10,
                   help="Số episode giữa mỗi lần in log của SB3")
    args = p.parse_args()
    train_dqn(args.timesteps, args.out, device=args.device, log_interval=args.log_interval)
    print(f"Saved model to {args.out}")
