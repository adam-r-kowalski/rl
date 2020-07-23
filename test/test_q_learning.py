import gym

import rl.q_learning as q_learning


def test_cross_entropy():
    env = gym.make('CartPole-v1')
    agent, episode = q_learning.agent(env)
    episodes = 1000
    for i in range(episodes):
        print(f'{i}/{episodes} = {episode(agent, env)}')
