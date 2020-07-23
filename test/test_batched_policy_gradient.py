import gym

import rl.batched_policy_gradient as batched_policy_gradient


def test_batched_policy_gradient():
    env = gym.make('CartPole-v1')
    agent, episode = batched_policy_gradient.agent(env)
    episodes = 400
    for i in range(episodes):
        print(f'{i}/{episodes} = {episode(agent, env)}')
