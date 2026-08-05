from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from scdsmdp.environment.types import Action, Adequacy, Fibrosis, Modality, Patient, Trajectory


class Policy(ABC):
    @abstractmethod
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        raise NotImplementedError


def action(interval: int, modality: Modality) -> int:
    return Action(interval, modality).index


class AASLDPolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        if patient.fibrosis == Fibrosis.F4:
            candidate = action(6, Modality.US_AFP)
        else:
            candidate = action(12, Modality.NO_SCREEN)
        return candidate if mask[candidate] else action(6, Modality.GALAD)


class EASLPolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        if patient.fibrosis >= Fibrosis.F3:
            candidate = action(6, Modality.US)
        else:
            candidate = action(12, Modality.NO_SCREEN)
        return candidate if mask[candidate] else action(6, Modality.AMRI)


class APASLPolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        high_risk = patient.fibrosis >= Fibrosis.F3 or patient.diabetes
        candidate = action(6 if high_risk else 12, Modality.US_AFP)
        return candidate if mask[candidate] else action(6, Modality.GALAD)


class FIB4Policy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        score = self._score(patient)
        if score >= 2.67:
            interval = 6
            modality = Modality.US_AFP
        elif score >= 1.3:
            interval = 12
            modality = Modality.US_AFP
        else:
            interval = 12
            modality = Modality.NO_SCREEN
        candidate = action(interval, modality)
        return candidate if mask[candidate] else action(interval, Modality.GALAD)

    @staticmethod
    def _score(patient: Patient) -> float:
        stage = (0.7, 1.0, 1.5, 2.5, 4.0)[int(patient.fibrosis)]
        trajectory = (0.8, 0.0, -0.4)[int(patient.trajectory)]
        return stage + trajectory + max(patient.age - 50.0, 0.0) / 40.0


class AMAPPolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        risk = 0.04 * patient.age + 0.7 * int(patient.fibrosis) + 0.5 * float(patient.diabetes)
        interval = 6 if risk >= 4.5 else 12
        modality = Modality.US_AFP if risk >= 3.0 else Modality.NO_SCREEN
        candidate = action(interval, modality)
        return candidate if mask[candidate] else action(interval, Modality.GALAD)


class GALADPolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        risk = int(patient.fibrosis) + float(patient.diabetes) + float(patient.age > 65)
        return action(6 if risk >= 4 else 12, Modality.GALAD)


class AMRIPolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        interval = 6 if patient.fibrosis >= Fibrosis.F3 else 12
        return action(interval, Modality.AMRI)


class RiskAdaptivePolicy(Policy):
    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        risk = self._risk(patient)
        interval = 3 if risk >= 7.0 else 6 if risk >= 4.0 else 12
        if patient.adequacy == Adequacy.C:
            modality = Modality.AMRI if risk >= 4.0 else Modality.GALAD
        elif risk >= 6.0:
            modality = Modality.AMRI
        elif risk >= 3.0:
            modality = Modality.GALAD
        else:
            modality = Modality.US_AFP
        candidate = action(interval, modality)
        if mask[candidate]:
            return candidate
        allowed = np.flatnonzero(mask)
        return int(allowed[0])

    @staticmethod
    def _risk(patient: Patient) -> float:
        score = 1.5 * int(patient.fibrosis)
        score += 1.4 if patient.trajectory == Trajectory.PROGRESSING else 0.0
        score += 0.8 if patient.diabetes else 0.0
        score += 0.6 if patient.bmi >= 30.0 else 0.0
        score += max(patient.age - 50.0, 0.0) / 25.0
        return score


class RandomPolicy(Policy):
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, patient: Patient, mask: NDArray[np.bool_]) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))


def registry(seed: int = 0) -> dict[str, Policy]:
    return {
        "aasld": AASLDPolicy(),
        "easl": EASLPolicy(),
        "apasl": APASLPolicy(),
        "fib4": FIB4Policy(),
        "amap": AMAPPolicy(),
        "galad": GALADPolicy(),
        "amri": AMRIPolicy(),
        "risk_adaptive": RiskAdaptivePolicy(),
        "random": RandomPolicy(seed),
    }
