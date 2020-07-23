import gym
import matplotlib.pyplot as plt

import rl.policy_gradient as policy_gradient
import rl.batched_policy_gradient as batched_policy_gradient
import rl.cross_entropy as cross_entropy
import rl.q_learning as q_learning
from rl.compare_agents import compare_agents


def cart_pole():
    return gym.make('CartPole-v1')


def acrobot():
    return gym.make('Acrobot-v1')


def test_compare_agents():
    agents = {'Policy Gradient': policy_gradient.agent,
              'Batched Policy Gradient': batched_policy_gradient.agent,
              'Cross Entropy': cross_entropy.agent,
              'Q Learning': q_learning.agent}
    agent_rewards = compare_agents(agents, acrobot, episodes=1000)
    plt.rc('font', size=20)
    for name, rewards in agent_rewards.items():
        plt.plot(rewards, label=name)
    plt.legend()
    plt.show()
