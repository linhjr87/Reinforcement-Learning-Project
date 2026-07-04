import time
import torch
from stable_baselines3 import DQN
from vasuki.gym_wrapper import GymWrapper
from vasuki.opponents import RandomOpponent, PolicySnapshotOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def _pick_device(device):
    # "auto" -> ưu tiên GPU nếu có; nếu chỉ định "cuda" mà không có GPU thì cảnh báo và về CPU
    if device in ("cuda", "auto") and not torch.cuda.is_available():
        if device == "cuda":
            print("[selfplay] CẢNH BÁO: yêu cầu 'cuda' nhưng không thấy GPU, chuyển sang CPU.")
        return "cpu"
    return device


def train_selfplay(total_timesteps=1_000_000, refresh_every=50_000,
                   save_path="models/dqn_selfplay.zip", seed=0, device="auto"):
    # Bắt đầu với đối thủ random; đóng băng snapshot A theo chu kỳ (fictitious self-play)
    device = _pick_device(device)
    env = GymWrapper(CONFIG, RandomOpponent())
    model = DQN("MlpPolicy", env, verbose=1, seed=seed,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=1_000, batch_size=64, gamma=0.95,
                device=device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"[selfplay] Thiết bị: {model.device} (GPU: {gpu_name}) | "
          f"timesteps={total_timesteps} | refresh={refresh_every} | seed={seed}")

    t0 = time.time()
    steps_done = 0
    while steps_done < total_timesteps:
        chunk = min(refresh_every, total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        # Lưu rồi load lại để tạo bản đóng băng ĐỘC LẬP (không dính weights đang cập nhật)
        model.save(save_path)
        snapshot = DQN.load(save_path)
        env.set_opponent(PolicySnapshotOpponent(snapshot))
        print(f"[selfplay] {steps_done}/{total_timesteps} steps | "
              f"đã làm mới snapshot đối thủ | {time.time() - t0:.1f}s")

    model.save(save_path)
    print(f"[selfplay] Hoàn tất sau {time.time() - t0:.1f}s. Đã lưu model vào {save_path}")
    return model
