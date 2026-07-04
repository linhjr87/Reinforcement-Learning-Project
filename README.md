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
Note: for these small MLP policies CPU is often faster than GPU (kernel-launch overhead
on tiny batches); `DEVICE = "auto"` uses the GPU when present.

## Results

Filled in after running the notebooks — see `summary.ipynb`.
