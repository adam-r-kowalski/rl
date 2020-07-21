from typing import Protocol
from abc import abstractmethod
from numpy import array

from rl.transition import Transition


class Agent(Protocol):
    @abstractmethod
    def select_action(self, obs: array) -> int:
        raise NotImplementedError

    @abstractmethod
    def store_transition(self, transition: Transition) -> None:
        raise NotImplementedError

    @abstractmethod
    def episode_start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def episode_end(self) -> None:
        raise NotImplementedError
