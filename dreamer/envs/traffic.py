import gym
import numpy as np
from rlpyt.envs.base import Env, EnvStep
from rlpyt.utils.collections import namedarraytuple
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
import yaml
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from dreamer.envs.env import EnvInfo

from wolf.utils.configuration.registry import R
AtariTrajInfo = AtariTrajInfo


class TrafficEnv(Env):
    def __init__(self, name, render, **kwargs):
        env_config_path = kwargs.get('env_config_path', None)
        env_name, env_config = self.load_config(env_config_path)
        self._size = tuple(env_config['agents_params']['params']['obs_params']['params']['size'])
        self._env = self.get_env(env_name, env_config)
        self._node_id = next(iter(self._env._agents.keys()))
        self.random = np.random.RandomState(seed=None)  # expose for one_hot wrapper

    def load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.load(file)

        experiments = config['ray']['run_experiments']['experiments']
        experiment = next(iter(experiments.values()))
        env_name = experiment['config']['env']
        env_config = experiment['config']['env_config']

        return env_name, env_config

    def get_env(self, env_name, env_config):
        create_env = R.env_factory(env_name)
        env = create_env(env_config)
        return env

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self._size, dtype="uint8")

    @property
    def action_space(self):
        # self._env.action_space() needs extra config for grouping agent action_spaces.
        return next(iter(self._env._agents.values())).action_space()

    def step(self, action):
        action = {self._node_id: action}
        obs, reward, done, info = self._env.step(action)
        self.obs = self.transform_obs(obs)
        done = done[self._node_id]
        reward = reward[self._node_id]
        info = info[self._node_id]

        info = EnvInfo(None, 0, done)
        return EnvStep(self.obs, reward, done, info)

    def transform_obs(self, obs):
        obs = obs[self._node_id]
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    def reset(self):
        obs = self._env.reset()
        self.obs = self.transform_obs(obs)
        return self.obs

    @property
    def horizon(self):
        raise NotImplementedError
