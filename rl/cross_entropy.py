from typing import List, Deque, Tuple
from collections import deque
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.distributions as distributions
import numpy as np
from numpy import array

from rl.env import Env
from rl.transition import Transition


@dataclass
class Episode:
    obs: List[array] = field(default_factory=list, init=False)
    action: List[int] = field(default_factory=list, init=False)


@dataclass
class ReplayMemory:
    episodes: Deque[Episode]
    rewards: Deque[float]
    percentile: float
    size: int

    def append(self, episode: Episode, reward: float) -> None:
        self.episodes.append(episode)
        self.rewards.append(reward)

    def batch(self) -> Tuple[Tensor, Tensor]:
        reward_bound = np.percentile(self.rewards, q=self.percentile)
        obs: List[array] = []
        actions: List[int] = []
        for i, reward in enumerate(self.rewards):
            if reward >= reward_bound:
                episode = self.episodes[i]
                obs += episode.obs
                actions += episode.action
        return torch.tensor(obs).float(), torch.tensor(actions)

    def __len__(self) -> int:
        return len(self.episodes)


@dataclass
class CrossEntropy:
    policy: nn.Module
    optimizer: Optimizer
    memory: ReplayMemory
    episode: Episode = field(default_factory=Episode, init=False)
    reward: float = field(default=0.0, init=False)

    def select_action(self, obs: array) -> int:
        obs = torch.from_numpy(obs).float()
        logits = self.policy(obs)
        m = distributions.Categorical(logits=logits)
        return int(m.sample())

    def store_transition(self, transition: Transition) -> None:
        self.episode.obs.append(transition.obs)
        self.episode.action.append(transition.action)
        self.reward += transition.reward

    def episode_start(self) -> None:
        pass

    def episode_end(self) -> None:
        self.memory.append(self.episode, self.reward)
        self.episode = Episode()
        self.reward = 0.0
        if len(self.memory) == self.memory.size:
            obs, actions = self.memory.batch()
            self.optimizer.zero_grad()
            logits = self.policy(obs)
            F.cross_entropy(logits, actions).backward()
            self.optimizer.step()


def cross_entropy(env: Env,
                  hidden_layers: List[int],
                  activation: nn.Module,
                  learning_rate: float,
                  memory_size: int,
                  percentile: float
                  ) -> CrossEntropy:
    layers: List[nn.Module] = []
    input_size = env.observation_space.shape[0]
    for hidden_layer in hidden_layers:
        layers.append(nn.Linear(input_size, hidden_layer))
        layers.append(activation)
        input_size = hidden_layer
    layers.append(nn.Linear(input_size, env.action_space.n))
    policy = nn.Sequential(*layers)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    memory = ReplayMemory(episodes=deque(maxlen=memory_size),
                          rewards=deque(maxlen=memory_size),
                          percentile=percentile,
                          size=memory_size)
    return CrossEntropy(policy, optimizer, memory)
