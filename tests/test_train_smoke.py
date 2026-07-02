import os
import numpy as np
from training.train_dqn import train_dqn


def test_train_dqn_smoke(tmp_path):
    save = tmp_path / "dqn_smoke.zip"
    model = train_dqn(total_timesteps=200, save_path=str(save), seed=0)
    assert os.path.exists(str(save))
    action, _ = model.predict(np.zeros(8, dtype=np.float32))
    assert int(action) in (0, 1, 2)
