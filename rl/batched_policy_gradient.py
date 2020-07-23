from typing import List
from dataclasses import dataclass, field
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.distributions as distributions
from numpy import array

from rl.env import Env
from rl.agent import Agent


@dataclass
class BatchedPolicyGradient:
    policy: nn.Module
    optimizer: Optimizer
    batch_size: int
    discount_factor: float
    rewards: List[float] = field(default_factory=list, init=False)
    log_probs: List[Tensor] = field(default_factory=list, init=False)
    losses: List[Tensor] = field(default_factory=list, init=False)


def discount(rewards: Tensor, gamma: float) -> Tensor:
    discounted = torch.zeros_like(rewards)
    running_sum = torch.tensor(0.0)
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        discounted[i] = running_sum
    return torch.tensor(discounted)


def normalize(rewards: Tensor) -> Tensor:
    return (rewards - torch.mean(rewards)) / torch.std(rewards)


def select_action(agent: BatchedPolicyGradient, obs: array) -> int:
    obs = torch.from_numpy(obs).float()
    probs = agent.policy(obs)
    m = distributions.Categorical(probs=probs)
    action = m.sample()
    agent.log_probs.append(m.log_prob(action))
    return int(action)


def improve_policy(agent: BatchedPolicyGradient) -> None:
    log_probs = torch.stack(agent.log_probs)
    discounted = discount(torch.tensor(agent.rewards), agent.discount_factor)
    returns = normalize(discounted)
    agent.losses.append(torch.sum(-log_probs * returns))
    agent.rewards = []
    agent.log_probs = []
    if len(agent.losses) >= agent.batch_size:
        torch.sum(torch.stack(agent.losses)).backward()
        agent.optimizer.step()
        agent.optimizer.zero_grad()
        agent.losses = []


def episode(agent: BatchedPolicyGradient, env: Env) -> float:
    done = False
    obs = env.reset()
    while not done:
        action = select_action(agent, obs)
        obs, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
    rewards = sum(agent.rewards)
    improve_policy(agent)
    return rewards


def agent(env: Env,
          hidden_layers: List[int] = [2**5, 2**5],
          activation: nn.Module = nn.LeakyReLU(),
          learning_rate: float = 1e-2,
          batch_size: int = 5,
          discount_factor: float = 0.99
          ) -> Agent[BatchedPolicyGradient]:
    layers: List[nn.Module] = []
    input_size = env.observation_space.shape[0]
    for hidden_layer in hidden_layers:
        layers.append(nn.Linear(input_size, hidden_layer))
        layers.append(activation)
        input_size = hidden_layer
    layers.append(nn.Linear(input_size, env.action_space.n))
    layers.append(nn.Softmax())
    policy = nn.Sequential(*layers)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    agent = BatchedPolicyGradient(policy,
                                  optimizer,
                                  batch_size,
                                  discount_factor)
    return agent, episode
