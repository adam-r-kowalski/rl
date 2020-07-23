from abc import abstractmethod
from typing import Dict, Tuple, List, Protocol, TypeVar
from multiprocessing import Queue, Process
import gym
import matplotlib.pyplot as plt

import rl.policy_gradient as policy_gradient
import rl.cross_entropy as cross_entropy
from rl.agent import MakeAgent


T = TypeVar('T')


class TypedQueue(Protocol[T]):
    @abstractmethod
    def get(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def put(self, t: T) -> None:
        raise NotImplementedError


def simulate(queue: TypedQueue[Tuple[str, float]],
             name: str,
             make_agent: MakeAgent,
             episodes: int
             ) -> None:
    env = gym.make('CartPole-v1')
    agent, episode = make_agent(env)
    for _ in range(episodes):
        queue.put((name, episode(agent, env)))


def finished(episodes: int, agent_rewards: Dict[str, List[float]]) -> bool:
    return all(len(rewards) == episodes for rewards in agent_rewards.values())


def compare_agents(agents: Dict[str, MakeAgent], episodes: int):
    agent_rewards: Dict[str, List[float]] = {
         name: [] for name in agents.keys()
    }
    queue: TypedQueue[Tuple[str, float]] = Queue()
    for name, module in agents.items():
        process = Process(target=simulate,
                          args=(queue, name, module, episodes))
        process.start()
    while not finished(episodes, agent_rewards):
        name, reward = queue.get()
        agent_rewards[name].append(reward)
        if len(agent_rewards[name]) % 10 == 0:
            for name, rewards in agent_rewards.items():
                if len(rewards) > 10:
                    last_ten_rewards = rewards[-10:]
                    avg_reward = sum(last_ten_rewards) / len(last_ten_rewards)
                    print(f'{name} {len(rewards)}/{episodes} {avg_reward:.2f}')
            print('\n')
    return agent_rewards


def test_compare_agents():
    agents = {'Policy Gradient': policy_gradient.agent,
              'Cross Entropy': cross_entropy.agent}
    agent_rewards = compare_agents(agents, episodes=1000)
    for name, rewards in agent_rewards.items():
        plt.plot(rewards, label=name)
    plt.legend()
    plt.show()
