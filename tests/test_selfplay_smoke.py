import os
from vasuki.selfplay import train_selfplay


def test_selfplay_smoke(tmp_path):
    save = tmp_path / "dqn_selfplay.zip"
    model = train_selfplay(total_timesteps=400, refresh_every=200,
                           save_path=str(save), seed=0)
    assert os.path.exists(str(save))
