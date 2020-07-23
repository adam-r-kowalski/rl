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
from rl.agent import Agent


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


@dataclass
class CrossEntropy:
    policy: nn.Module
    optimizer: Optimizer
    memory: ReplayMemory
    episode: Episode = field(default_factory=Episode, init=False)
    reward: float = field(default=0.0, init=False)


def remember(memory: ReplayMemory, episode: Episode, reward: float) -> None:
    memory.episodes.append(episode)
    memory.rewards.append(reward)


def batch(memory: ReplayMemory) -> Tuple[Tensor, Tensor]:
    reward_bound = np.percentile(memory.rewards, q=memory.percentile)
    obs: List[array] = []
    actions: List[int] = []
    for i, reward in enumerate(memory.rewards):
        if reward >= reward_bound:
            episode = memory.episodes[i]
            obs += episode.obs
            actions += episode.action
    return torch.tensor(obs).float(), torch.tensor(actions)


def select_action(agent: CrossEntropy, obs: array) -> int:
    obs = torch.from_numpy(obs).float()
    logits = agent.policy(obs)
    m = distributions.Categorical(logits=logits)
    return int(m.sample())


def improve_policy(agent: CrossEntropy) -> None:
    if len(agent.memory.episodes) < agent.memory.size:
        return
    obs, actions = batch(agent.memory)
    agent.optimizer.zero_grad()
    logits = agent.policy(obs)
    F.cross_entropy(logits, actions).backward()
    agent.optimizer.step()


def episode(agent: CrossEntropy, env: Env, render: bool = False) -> float:
    done = False
    obs = env.reset()
    while not done:
        action = select_action(agent, obs)
        next_obs, reward, done, _ = env.step(action)
        agent.episode.obs.append(obs)
        agent.episode.action.append(action)
        agent.reward += reward
        obs = next_obs
        if render:
            env.render()
    reward = agent.reward
    remember(agent.memory, agent.episode, agent.reward)
    agent.episode = Episode()
    agent.reward = 0.0
    improve_policy(agent)
    return reward


def agent(env: Env,
          hidden_layers: List[int] = [2**5, 2**5],
          activation: nn.Module = nn.LeakyReLU(),
          learning_rate: float = 1e-2,
          memory_size: int = 20,
          percentile: float = 70.0
          ) -> Agent[CrossEntropy]:
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
    return CrossEntropy(policy, optimizer, memory), episode
