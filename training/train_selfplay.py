import argparse
from vasuki.selfplay import train_selfplay

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--refresh", type=int, default=50_000)
    p.add_argument("--out", type=str, default="models/dqn_selfplay.zip")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cpu (auto sẽ dùng GPU nếu có)")
    args = p.parse_args()
    train_selfplay(args.timesteps, args.refresh, args.out, device=args.device)
    print(f"Saved self-play model to {args.out}")
