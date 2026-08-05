from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scdsmdp.environment.economics import HealthEconomics
from scdsmdp.environment.natural_history import NaturalHistory
from scdsmdp.environment.population import PopulationSampler
from scdsmdp.environment.screening import ScreeningModel
from scdsmdp.environment.types import (
    Action,
    CancerStage,
    EpisodeSummary,
    Patient,
)
from scdsmdp.specification import ExperimentSpec


@dataclass(frozen=True)
class StepResult:
    observation: NDArray[np.float32]
    reward: float
    terminated: bool
    constraints: NDArray[np.float32]
    mask: NDArray[np.bool_]


class SurveillanceSimulator:
    def __init__(self, spec: ExperimentSpec, seed: int | None = None) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed if seed is None else seed)
        self.population = PopulationSampler()
        self.history = NaturalHistory()
        self.screening = ScreeningModel()
        self.economics = HealthEconomics(spec.simulator.economic_discount)
        self.patient: Patient | None = None
        self.quarter = 0
        self.hcc_occurred = False
        self.early_detected = False

    def reset(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        self.patient = self.population.sample(self.rng)
        self.quarter = 0
        self.hcc_occurred = False
        self.early_detected = False
        return self.patient.vector(), self.screening.action_mask(self.patient)

    def step(self, action_index: int) -> StepResult:
        patient = self._patient()
        action = Action.from_index(action_index)
        mask = self.screening.action_mask(patient)
        if not mask[action_index]:
            raise ValueError("masked action selected")
        before_cost = patient.accumulated_cost
        before_cancer = patient.cancer
        due = patient.months_since_screen >= action.interval_months or self.quarter == 0
        detected = False
        false_positive = False
        if due:
            detected, false_positive = self.screening.observe(patient, action, self.rng)
        self.history.advance(patient, self.rng)
        if patient.cancer != CancerStage.NONE:
            self.hcc_occurred = True
        if detected and before_cancer == CancerStage.EARLY:
            self.early_detected = True
        qaly = self.economics.quarterly_qaly(patient, self.quarter)
        cost = patient.accumulated_cost - before_cost
        reward = self._reward(patient, detected, false_positive, qaly, cost)
        constraints = self._constraints(patient, action)
        self.quarter += 1
        horizon = self.spec.simulator.horizon_years * self.spec.simulator.decisions_per_year
        terminated = not patient.alive or patient.detected or self.quarter >= horizon
        if terminated and patient.cancer != CancerStage.NONE and not patient.detected:
            patient.missed = True
        return StepResult(
            patient.vector(),
            reward,
            terminated,
            constraints,
            self.screening.action_mask(patient),
        )

    def _reward(
        self,
        patient: Patient,
        detected: bool,
        false_positive: bool,
        qaly: float,
        cost: float,
    ) -> float:
        weights = self.spec.reward
        detection = float(detected and patient.cancer == CancerStage.EARLY)
        burden = float(false_positive)
        if patient.cancer == CancerStage.ADVANCED and detected:
            burden += 1.0
        return (
            weights.detection * detection
            + weights.qaly * qaly
            - weights.cost * (cost / 1000.0)
            - weights.burden * burden
        )

    def _constraints(self, patient: Patient, action: Action) -> NDArray[np.float32]:
        missed = float(patient.cancer != CancerStage.NONE and not patient.detected)
        maximum = 6 if int(patient.fibrosis) >= 3 else 12
        interval = float(max(0, action.interval_months - maximum))
        annual_cost = patient.accumulated_cost / max((self.quarter + 1) / 4.0, 0.25)
        budget = max(0.0, annual_cost - self.spec.constraints.annual_budget)
        return np.asarray((missed, interval, budget), dtype=np.float32)

    def summary(self) -> EpisodeSummary:
        patient = self._patient()
        return EpisodeSummary(
            self.hcc_occurred,
            self.early_detected,
            patient.detected,
            patient.missed,
            patient.screens,
            patient.accumulated_cost,
            patient.accumulated_qaly,
            self.quarter / 4.0,
            patient.false_positives,
        )

    def _patient(self) -> Patient:
        if self.patient is None:
            raise RuntimeError("reset must be called before use")
        return self.patient
