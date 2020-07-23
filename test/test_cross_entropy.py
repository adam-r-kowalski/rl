import gym

import rl.cross_entropy as cross_entropy


def test_cross_entropy():
    env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')
    agent, episode = cross_entropy.agent(env)
    episodes = 400
    for i in range(episodes):
        print(f'{i}/{episodes} = {episode(agent, env, render=True)}')
