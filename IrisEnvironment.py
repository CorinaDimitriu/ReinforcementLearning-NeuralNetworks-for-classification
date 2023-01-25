import random
from abc import ABC
import gym
import numpy as np


class IrisEnvironment(gym.Env, ABC):
    def __init__(self, dataset, images_per_episode=1, train=True):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(4,),
                                                dtype=np.float32)
        self.images_per_episode = images_per_episode
        self.step_count = 0
        self.train = train
        self.x, self.y = dataset
        self.dataset_idx = -1

    def step(self, action):  # action is part of the action_space
        reward = int(action == self.expected_action)
        obs = self._next_obs()
        self.step_count += 1
        done = self.step_count >= self.images_per_episode
        return obs, reward, done, {}

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self._next_obs()

    def _next_obs(self):
        if self.train:
            no_obs = random.randint(0, len(self.x) - 1)
        else:
            self.dataset_idx += 1
            no_obs = self.dataset_idx
        if self.dataset_idx >= len(self.x):
            raise StopIteration()
        obs = self.x[no_obs]
        self.expected_action = int(self.y[no_obs])
        return obs
