import gym
import torch.nn as nn

from rl.policy_gradient import policy_gradient
from rl.simulate import simulate
import rl.monitor as monitor


def test_policy_gradient():
    env = gym.make('CartPole-v1')
    agent = policy_gradient(env,
                            hidden_layers=[20, 20],
                            activation=nn.LeakyReLU(),
                            learning_rate=0.01)
    simulate(env, agent,
             monitor.Compose([monitor.Progress(),
                              monitor.Plot(),
                              monitor.Render()]),
             episodes=400)
