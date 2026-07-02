import os
from training.train_tabular import train_tabular


def test_tabular_smoke(tmp_path):
    save = tmp_path / "qtable_smoke.pkl"
    q = train_tabular(num_episodes=2, save_path=str(save), seed=0)
    assert os.path.exists(str(save))
    assert len(q) > 0
