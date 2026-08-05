from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from scdsmdp.environment.types import Transition


@dataclass(frozen=True)
class Batch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    constraints: Tensor
    next_masks: Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, action_size: int, seed: int) -> None:
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)
        self.observations = np.empty((capacity, state_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, state_size), dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.constraints = np.empty((capacity, 3), dtype=np.float32)
        self.next_masks = np.empty((capacity, action_size), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def append(self, transition: Transition) -> None:
        index = self.position
        self.observations[index] = transition.observation
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.next_observations[index] = transition.next_observation
        self.terminated[index] = float(transition.terminated)
        self.constraints[index] = transition.constraints
        self.next_masks[index] = transition.next_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        if batch_size > self.size:
            raise ValueError("batch exceeds stored transitions")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return Batch(
            self._tensor(self.observations[indices], device),
            self._tensor(self.actions[indices], device, torch.long),
            self._tensor(self.rewards[indices], device),
            self._tensor(self.next_observations[indices], device),
            self._tensor(self.terminated[indices], device),
            self._tensor(self.constraints[indices], device),
            self._tensor(self.next_masks[indices], device, torch.bool),
        )

    @staticmethod
    def _tensor(
        values: NDArray[np.generic],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return torch.as_tensor(values, device=device, dtype=dtype)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "position": self.position,
            "size": self.size,
            "observations": self.observations[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_observations": self.next_observations[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "constraints": self.constraints[: self.size].copy(),
            "next_masks": self.next_masks[: self.size].copy(),
        }
