from dataclasses import dataclass
from numpy import array


@dataclass
class Transition:
    obs: array
    action: int
    reward: float
    next_obs: array
    done: bool
