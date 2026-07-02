import argparse
from vasuki.selfplay import train_selfplay

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--refresh", type=int, default=50_000)
    p.add_argument("--out", type=str, default="models/dqn_selfplay.zip")
    args = p.parse_args()
    train_selfplay(args.timesteps, args.refresh, args.out)
    print(f"Saved self-play model to {args.out}")
