from typing import TypeVar, Callable, Tuple

from rl.env import Env

T = TypeVar('T')
Episode = Callable[[T, Env], float]
Agent = Tuple[T, Episode[T]]
MakeAgent = Callable[[Env], Agent]
