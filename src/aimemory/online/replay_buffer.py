"""Experience Replay Buffer for online RL training."""

import collections
import pickle
from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None


class ReplayBuffer:
    """Circular buffer for experience replay."""

    def __init__(self, capacity: int = 5000):
        self._buffer: collections.deque[Experience] = collections.deque(maxlen=capacity)
        self._capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
    ) -> None:
        self._buffer.append(Experience(state, action, reward, next_state))

    def sample(self, batch_size: int = 32) -> list[Experience]:
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size ({batch_size}) exceeds buffer size ({len(self._buffer)})."
            )
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        buf_list = list(self._buffer)
        return [buf_list[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._buffer, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._buffer = pickle.load(f)
