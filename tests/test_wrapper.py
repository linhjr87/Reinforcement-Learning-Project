from vasuki.gym_wrapper import GymWrapper
from vasuki.opponents import RandomOpponent

CONFIG = {"n": 8, "rewards": {"Food": 4, "Movement": -1, "Illegal": -2}, "game_length": 100}


def test_reset_returns_obs_info():
    env = GymWrapper(CONFIG, RandomOpponent())
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    assert isinstance(info, dict)


def test_step_returns_five_values():
    env = GymWrapper(CONFIG, RandomOpponent())
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (8,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_sb3_check_env():
    from stable_baselines3.common.env_checker import check_env
    env = GymWrapper(CONFIG, RandomOpponent())
    check_env(env, warn=True)  # không raise
