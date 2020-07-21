from typing import List
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from numpy import array

from rl.env import Env
from rl.transition import Transition


def discount(rewards: Tensor, gamma: float) -> Tensor:
    discounted = torch.zeros_like(rewards)
    running_sum = torch.tensor(0.0)
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + gamma * running_sum
        discounted[i] = running_sum
    return torch.tensor(discounted)


def normalize(rewards: Tensor) -> Tensor:
    return (rewards - torch.mean(rewards)) / torch.std(rewards)


@dataclass
class PolicyGradient:
    policy: nn.Module
    optimizer: optim.Optimizer
    rewards: List[float]
    log_probs: List[Tensor]

    def select_action(self, obs: array) -> int:
        obs = torch.from_numpy(obs).float()
        probs = self.policy(obs)
        m = distributions.Categorical(probs=probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return int(action)

    def store_transition(self, transition: Transition) -> None:
        self.rewards.append(transition.reward)

    def episode_start(self) -> None:
        self.optimizer.zero_grad()

    def episode_end(self) -> None:
        log_probs = torch.stack(self.log_probs)
        returns = normalize(discount(torch.tensor(self.rewards), gamma=0.9))
        torch.sum(-log_probs * returns).backward()
        self.optimizer.step()
        self.rewards = []
        self.log_probs = []


def policy_gradient(env: Env,
                    hidden_layers: List[int],
                    activation: nn.Module,
                    learning_rate: float
                    ) -> PolicyGradient:
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
    return PolicyGradient(policy, optimizer, rewards=[], log_probs=[])
