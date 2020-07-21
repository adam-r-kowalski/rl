from typing import Protocol, List
from dataclasses import dataclass, field
from abc import abstractmethod
import matplotlib.pyplot as plt

from rl.transition import Transition
from rl.env import Env


class Monitor(Protocol):
    @abstractmethod
    def store_transition(self, env: Env, transition: Transition) -> None:
        raise NotImplementedError

    @abstractmethod
    def episode_end(self, episode: int, episodes: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def simulation_end(self) -> None:
        raise NotImplementedError


@dataclass
class Compose:
    monitors: List[Monitor]

    def store_transition(self, env: Env, transition: Transition) -> None:
        for monitor in self.monitors:
            monitor.store_transition(env, transition)

    def episode_end(self, episode: int, episodes: int) -> None:
        for monitor in self.monitors:
            monitor.episode_end(episode, episodes)

    def simulation_end(self) -> None:
        for monitor in self.monitors:
            monitor.simulation_end()


@dataclass
class Progress:
    rewards: List[float] = field(default_factory=list, init=False)

    def store_transition(self, env: Env, transition: Transition) -> None:
        self.rewards.append(transition.reward)

    def episode_end(self, episode: int, episodes: int) -> None:
        print(f'Episode {episode}/{episodes} Reward {sum(self.rewards)}')
        self.rewards = []

    def simulation_end(self) -> None:
        print('Simulation Over')


@dataclass
class Plot:
    rewards: List[float] = field(default_factory=list, init=False)
    accumulator: float = field(default=0.0, init=False)

    def store_transition(self, env: Env, transition: Transition) -> None:
        self.accumulator += transition.reward

    def episode_end(self, episode: int, episodes: int) -> None:
        self.rewards.append(self.accumulator)
        self.accumulator = 0.0

    def simulation_end(self) -> None:
        plt.plot(self.rewards)
        plt.show()


class Render:
    def store_transition(self, env: Env, transition: Transition) -> None:
        env.render()

    def episode_end(self, episode: int, episodes: int) -> None:
        pass

    def simulation_end(self) -> None:
        pass


@dataclass
class RenderBest:
    best: float = field(default=0.0, init=False)
    should_render: bool = field(default=True, init=False)
    rewards: List[float] = field(default_factory=list, init=False)

    def store_transition(self, env: Env, transition: Transition) -> None:
        if self.should_render:
            env.render()
        self.rewards.append(transition.reward)

    def episode_end(self, episode: int, episodes: int) -> None:
        rewards = sum(self.rewards)
        self.rewards = []
        if rewards >= self.best:
            self.should_render = True
            self.best = rewards
        else:
            self.should_render = False

    def simulation_end(self) -> None:
        pass
