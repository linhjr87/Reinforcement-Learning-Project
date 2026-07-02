import numpy as np
from vasuki.features import get_input_states, to_vector


class RandomOpponent:
    def act(self, env):
        return int(env.action_space.sample())


class TabularOpponent:
    def __init__(self, q_table):
        self.q_table = q_table

    def act(self, env):
        obs = get_input_states(env, player="A")
        if obs not in self.q_table:
            return int(env.action_space.sample())
        return int(np.argmax(self.q_table[obs]))


class PolicySnapshotOpponent:
    def __init__(self, model):
        self.model = model

    def act(self, env):
        obs = to_vector(env, player="A")
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
