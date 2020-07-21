from abc import abstractmethod
from typing import Protocol, Tuple, Sequence
from numpy import ndarray


class Discrete(Protocol):
    @property
    @abstractmethod
    def n(self) -> int:
        raise NotImplementedError


class Box(Protocol):
    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        raise NotImplementedError


class Env(Protocol):
    @abstractmethod
    def reset(self) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> Tuple[ndarray, float, bool, dict]:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> Discrete:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
