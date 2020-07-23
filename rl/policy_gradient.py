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
class PolicyGradient:
    policy: nn.Module
    optimizer: Optimizer
    rewards: List[float] = field(default_factory=list, init=False)
    log_probs: List[Tensor] = field(default_factory=list, init=False)


def discount(rewards: Tensor, gamma: float) -> Tensor:
    discounted = torch.zeros_like(rewards)
    running_sum = torch.tensor(0.0)
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        discounted[i] = running_sum
    return torch.tensor(discounted)


def normalize(rewards: Tensor) -> Tensor:
    return (rewards - torch.mean(rewards)) / torch.std(rewards)


def select_action(agent: PolicyGradient, obs: array) -> int:
    obs = torch.from_numpy(obs).float()
    probs = agent.policy(obs)
    m = distributions.Categorical(probs=probs)
    action = m.sample()
    agent.log_probs.append(m.log_prob(action))
    return int(action)


def improve_policy(agent: PolicyGradient) -> None:
    log_probs = torch.stack(agent.log_probs)
    returns = normalize(discount(torch.tensor(agent.rewards), gamma=0.9))
    torch.sum(-log_probs * returns).backward()
    agent.optimizer.step()
    agent.rewards = []
    agent.log_probs = []


def episode(agent: PolicyGradient, env: Env) -> float:
    agent.optimizer.zero_grad()
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
          hidden_layers: List[int] = [20, 20],
          activation: nn.Module = nn.LeakyReLU(),
          learning_rate: float = 0.01,
          discount_factor: float = 0.9
          ) -> Agent[PolicyGradient]:
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
    return PolicyGradient(policy, optimizer), episode
