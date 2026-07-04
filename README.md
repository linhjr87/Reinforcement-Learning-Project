# Vasuki RL — Four Research Directions

Standalone experiment branch for the Vasuki two-snake grid environment. Four research
directions, each in one fully self-contained notebook (no imports from the parent repo):

| Notebook | Question |
|---|---|
| `direction1_selfplay_league.ipynb` | Does an opponent pool beat single-snapshot fictitious self-play? |
| `direction2_representation.ipynb` | Hand-crafted features vs full-grid observation: learning speed and transfer |
| `direction3_reward_shaping.ipynb` | How do collision rewards shape emergent behavior? |
| `direction4_pomdp_elo.ipynb` | Is the missing enemy head direction a bottleneck? + Elo ranking of all agents |
| `summary.ipynb` | Cross-direction comparison from saved results (no training) |

## How to run

```bash
pip install -r requirements.txt
jupyter lab   # run notebooks top-to-bottom from this directory
```

Order: notebooks 1–3 are independent; notebook 4's Elo section consumes their saved
agents from `artifacts/` (missing ones are skipped with a warning); `summary.ipynb` last.

Each notebook's config cell has course-scale defaults with full-scale values in comments.
The config cell resolves `DEVICE = "cuda"` when a CUDA GPU is available (falling back to
`"cpu"`) and prints the device it trains on, so DQN training runs on the GPU when present.
Note: for these small MLP policies the CPU is often as fast or faster than the GPU
(kernel-launch overhead on tiny batches dominates, and the env loop is CPU-bound); set
`DEVICE = "cpu"` in the config cell if you find it quicker, especially at course scale.

## Results

This branch ships with `results/` empty (only a `.gitkeep`) — result JSONs are generated
locally, not committed. To regenerate: run notebooks 1 -> 4 top-to-bottom (each writes its
own `results/direction<N>_*.json`; direction 4 additionally writes `results/elo_ratings.json`),
then run `summary.ipynb` last to aggregate everything into a master table and an Elo ranking
chart with confidence intervals.
