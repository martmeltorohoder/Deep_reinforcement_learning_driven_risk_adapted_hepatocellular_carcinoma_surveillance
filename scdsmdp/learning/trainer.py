from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scdsmdp.environment.simulator import SurveillanceSimulator
from scdsmdp.environment.types import Transition
from scdsmdp.learning.agent import SCDSMDPAgent, UpdateResult
from scdsmdp.learning.replay import ReplayBuffer
from scdsmdp.specification import ExperimentSpec


@dataclass(frozen=True)
class TrainingProgress:
    episode: int
    step: int
    mean_return: float
    mean_length: float
    epsilon: float
    update: UpdateResult | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, spec: ExperimentSpec, output: Path, device: torch.device) -> None:
        set_seed(spec.seed)
        self.spec = spec
        self.output = output
        self.device = device
        self.logger = logging.getLogger("scdsmdp.training")
        self.environment = SurveillanceSimulator(spec)
        self.agent = SCDSMDPAgent(spec, device)
        self.replay = ReplayBuffer(
            spec.training.replay_capacity,
            spec.model.state_size,
            spec.model.action_size,
            spec.seed,
        )
        self.steps = 0
        self.start_episode = 0

    def train(self) -> list[TrainingProgress]:
        self.output.mkdir(parents=True, exist_ok=True)
        reports: list[TrainingProgress] = []
        returns: list[float] = []
        lengths: list[int] = []
        latest: UpdateResult | None = None
        for episode in range(self.start_episode, self.spec.training.episodes):
            observation, mask = self.environment.reset()
            terminated = False
            episode_return = 0.0
            episode_length = 0
            while not terminated:
                epsilon = self._epsilon(self.steps)
                action = self.agent.act(observation, mask, epsilon, self.environment.rng)
                result = self.environment.step(action)
                self.replay.append(
                    Transition(
                        observation,
                        action,
                        result.reward,
                        result.observation,
                        result.terminated,
                        result.constraints,
                        result.mask,
                    )
                )
                observation = result.observation
                mask = result.mask
                terminated = result.terminated
                episode_return += result.reward
                episode_length += 1
                self.steps += 1
                if self._ready():
                    batch = self.replay.sample(self.spec.training.batch_size, self.device)
                    latest = self.agent.update(batch, self.steps)
            returns.append(episode_return)
            lengths.append(episode_length)
            if (episode + 1) % self.spec.training.evaluation_frequency == 0:
                progress = TrainingProgress(
                    episode + 1,
                    self.steps,
                    float(np.mean(returns[-1000:])),
                    float(np.mean(lengths[-1000:])),
                    self._epsilon(self.steps),
                    latest,
                )
                reports.append(progress)
                self._record(progress)
                self.save(self.output / "latest.pt", episode + 1)
        self.save(self.output / "final.pt", self.spec.training.episodes)
        return reports

    def _ready(self) -> bool:
        return (
            self.replay.size
            >= max(self.spec.training.learning_starts, self.spec.training.batch_size)
            and self.steps % self.spec.training.train_frequency == 0
        )

    @staticmethod
    def _epsilon(step: int) -> float:
        fraction = min(step / 250000.0, 1.0)
        return 1.0 + fraction * (0.02 - 1.0)

    def _record(self, progress: TrainingProgress) -> None:
        payload = asdict(progress)
        with (self.output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self.logger.info(
            "episode=%d step=%d return=%.4f epsilon=%.4f",
            progress.episode,
            progress.step,
            progress.mean_return,
            progress.epsilon,
        )

    def save(self, destination: Path, episode: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        state: dict[str, Any] = {
            "agent": self.agent.state_dict(),
            "episode": episode,
            "steps": self.steps,
            "seed": self.spec.seed,
            "numpy_rng": self.environment.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }
        torch.save(state, temporary)
        os.replace(temporary, destination)

    def resume(self, source: Path) -> None:
        state = torch.load(source, map_location=self.device, weights_only=False)
        if int(state["seed"]) != self.spec.seed:
            raise ValueError("checkpoint seed differs from configuration")
        self.agent.load_state_dict(state["agent"])
        self.start_episode = int(state["episode"])
        self.steps = int(state["steps"])
        self.environment.rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
