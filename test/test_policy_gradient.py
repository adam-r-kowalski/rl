import gym

import rl.policy_gradient as policy_gradient


def test_policy_gradient():
    env = gym.make('CartPole-v1')
    agent, episode = policy_gradient.agent(env)
    episodes = 400
    for i in range(episodes):
        print(f'{i}/{episodes} = {episode(agent, env)}')
