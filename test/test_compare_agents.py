from multiprocessing import Queue, Process
import gym
import matplotlib.pyplot as plt

import rl.policy_gradient as policy_gradient
import rl.cross_entropy as cross_entropy


def simulate(queue: Queue, name: str, module, episodes: int) -> None:
    env = gym.make('CartPole-v1')
    agent, episode = module.agent(env)
    for _ in range(episodes):
        queue.put((name, episode(agent, env)))


def test_compare_agents():
    agents = {'Policy Gradient': policy_gradient,
              'Cross Entropy': cross_entropy}
    agent_rewards = {name: [] for name in agents.keys()}
    episodes = 1000
    queue = Queue()
    for name, module in agents.items():
        process = Process(target=simulate,
                          args=(queue, name, module, episodes))
        process.start()

    def finished() -> bool:
        lens = (len(rewards) == episodes for rewards in agent_rewards.values())
        return all(lens)

    while not finished():
        name, reward = queue.get()
        agent_rewards[name].append(reward)
        if len(agent_rewards[name]) % 10 == 0:
            for name, rewards in agent_rewards.items():
                print(f'{name} {len(rewards)}/{episodes}')
            print('\n')

    for name, rewards in agent_rewards.items():
        plt.plot(rewards, label=name)
    plt.legend()
    plt.show()
