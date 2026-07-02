import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from vasuki.env import Vasuki
from vasuki.features import to_vector


class GymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config, opponent):
        super().__init__()
        self.config = config
        self.raw_env = Vasuki(**config)
        self.opponent = opponent
        self.observation_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = Discrete(3)

    def set_opponent(self, opponent):
        self.opponent = opponent

    def _opponent_action(self):
        # Tráo tạm để opponent (vốn quyết định như "A") điều khiển agent B
        self.raw_env.agentA, self.raw_env.agentB = self.raw_env.agentB, self.raw_env.agentA
        try:
            return self.opponent.act(self.raw_env)
        finally:
            self.raw_env.agentA, self.raw_env.agentB = self.raw_env.agentB, self.raw_env.agentA

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.raw_env.reset()
        obs = to_vector(self.raw_env, player="A")
        return obs, {}

    def step(self, action):
        action_b = self._opponent_action()
        rewardA, rewardB, done, info = self.raw_env.step_two(int(action), action_b)
        obs = to_vector(self.raw_env, player="A")
        terminated = bool(done)
        truncated = False
        return obs, float(rewardA), terminated, truncated, info
