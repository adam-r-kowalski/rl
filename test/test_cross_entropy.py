import gym
import torch.nn as nn

from rl.cross_entropy import cross_entropy
from rl.simulate import simulate
import rl.monitor as monitor


def test_cross_entropy():
    env = gym.make('CartPole-v1')
    agent = cross_entropy(env,
                          hidden_layers=[20, 20],
                          activation=nn.LeakyReLU(),
                          learning_rate=0.01,
                          memory_size=20,
                          percentile=70)
    simulate(env, agent,
             monitor.Compose([monitor.Progress(),
                              monitor.Plot(),
                              monitor.Render()]),
             episodes=400)
