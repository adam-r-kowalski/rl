from typing import List, Deque
from collections import deque
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from numpy import array

from rl.env import Env
from rl.agent import Agent


@dataclass
class ReplayMemory:
    obs: Deque[array]
    action: Deque[int]
    reward: Deque[float]
    next_obs: Deque[array]
    done: Deque[bool]


@dataclass
class QLearning:
    action_space: int
    policy: nn.Module
    optimizer: Optimizer
    memory: ReplayMemory
    batch_size: int
    discount_factor: float
    epsilon_decay: float
    epsilon_min: float
    epsilon: float = field(default=1.0, init=False)


@dataclass
class MiniBatch:
    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    done: Tensor


def select_action(agent: QLearning, obs: array) -> int:
    obs = torch.from_numpy(obs).float()
    if torch.rand(size=(1,)) < agent.epsilon:
        return int(torch.randint(0, agent.action_space, (1,)))
    logits = agent.policy(obs)
    return int(torch.argmax(logits))


def mini_batch(agent: QLearning) -> MiniBatch:
    indices = torch.randint(0, len(agent.memory.obs), (agent.batch_size,))
    obs = torch.tensor(agent.memory.obs).float()[indices]
    action = torch.tensor(agent.memory.action)[indices]
    reward = torch.tensor(agent.memory.reward)[indices]
    next_obs = torch.tensor(agent.memory.next_obs).float()[indices]
    done = torch.tensor(agent.memory.done)[indices]
    return MiniBatch(obs, action, reward, next_obs, done)


def improve(agent: QLearning) -> None:
    if len(agent.memory.obs) < agent.batch_size:
        return
    agent.optimizer.zero_grad()
    batch = mini_batch(agent)
    y_hat = agent.policy(batch.obs).gather(1, batch.action.view((-1, 1)))
    next_q = agent.policy(batch.next_obs).max(1)[0].detach()
    next_q[batch.done] = 0
    y = batch.reward + agent.discount_factor * next_q
    F.smooth_l1_loss(y_hat, y).backward()
    agent.optimizer.step()
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)


def episode(agent: QLearning, env: Env) -> float:
    done = False
    obs = env.reset()
    episode_reward = 0.0
    while not done:
        action = select_action(agent, obs)
        next_obs, reward, done, _ = env.step(action)
        agent.memory.obs.append(obs)
        agent.memory.action.append(action)
        agent.memory.reward.append(reward)
        agent.memory.next_obs.append(next_obs)
        agent.memory.done.append(done)
        episode_reward += reward
        obs = next_obs
    improve(agent)
    return episode_reward


def agent(env: Env,
          hidden_layers: List[int] = [2**5, 2**5],
          activation: nn.Module = nn.LeakyReLU(),
          learning_rate: float = 1e-2,
          memory_size: int = 100,
          batch_size: int = 10,
          discount_factor: float = 0.9,
          epsilon_decay: float = 0.99,
          epsilon_min: float = 0.01
          ) -> Agent[QLearning]:
    layers: List[nn.Module] = []
    input_size = env.observation_space.shape[0]
    for hidden_layer in hidden_layers:
        layers.append(nn.Linear(input_size, hidden_layer))
        layers.append(activation)
        input_size = hidden_layer
    layers.append(nn.Linear(input_size, env.action_space.n))
    policy = nn.Sequential(*layers)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    memory = ReplayMemory(obs=deque(maxlen=memory_size),
                          action=deque(maxlen=memory_size),
                          reward=deque(maxlen=memory_size),
                          next_obs=deque(maxlen=memory_size),
                          done=deque(maxlen=memory_size))
    agent = QLearning(env.action_space.n,
                      policy,
                      optimizer,
                      memory,
                      batch_size,
                      discount_factor,
                      epsilon_decay,
                      epsilon_min)
    return agent, episode
