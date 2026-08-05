from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from scdsmdp.assessment.metrics import SurveillanceMetrics, aggregate
from scdsmdp.environment.simulator import SurveillanceSimulator
from scdsmdp.environment.types import Patient
from scdsmdp.specification import ExperimentSpec

PolicyFunction = Callable[[NDArray[np.float32], NDArray[np.bool_], Patient], int]


class Evaluator:
    def __init__(self, spec: ExperimentSpec, seed: int) -> None:
        self.simulator = SurveillanceSimulator(spec, seed)

    def evaluate(self, policy: PolicyFunction, episodes: int) -> SurveillanceMetrics:
        summaries = []
        for _ in range(episodes):
            observation, mask = self.simulator.reset()
            terminated = False
            while not terminated:
                patient = self.simulator.patient
                if patient is None:
                    raise RuntimeError("simulator patient unavailable")
                selected = policy(observation, mask, patient)
                result = self.simulator.step(selected)
                observation = result.observation
                mask = result.mask
                terminated = result.terminated
            summaries.append(self.simulator.summary())
        return aggregate(summaries)
