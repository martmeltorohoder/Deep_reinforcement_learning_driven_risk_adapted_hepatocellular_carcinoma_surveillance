from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Fibrosis(IntEnum):
    F0 = 0
    F1 = 1
    F2 = 2
    F3 = 3
    F4 = 4


class Trajectory(IntEnum):
    PROGRESSING = 0
    STABLE = 1
    REGRESSING = 2


class Adequacy(IntEnum):
    A = 0
    B = 1
    C = 2


class Modality(IntEnum):
    US = 0
    US_AFP = 1
    GALAD = 2
    AMRI = 3
    LIQUID_BIOPSY = 4
    NO_SCREEN = 5


class CancerStage(IntEnum):
    NONE = 0
    EARLY = 1
    ADVANCED = 2


@dataclass(frozen=True)
class Action:
    interval_months: int
    modality: Modality

    @property
    def index(self) -> int:
        return (
            self.interval_months // 3 - 1
            if self.interval_months == 3
            else {6: 1, 12: 2}[self.interval_months]
        ) * 6 + int(self.modality)

    @classmethod
    def from_index(cls, index: int) -> Action:
        if not 0 <= index < 18:
            raise ValueError("action index must be in [0, 18)")
        return cls((3, 6, 12)[index // 6], Modality(index % 6))


@dataclass
class Patient:
    fibrosis: Fibrosis
    trajectory: Trajectory
    bmi: float
    age: float
    diabetes: bool
    adequacy: Adequacy
    months_since_screen: int = 0
    last_result: int = 0
    cancer: CancerStage = CancerStage.NONE
    alive: bool = True
    decompensated: bool = False
    accumulated_cost: float = 0.0
    accumulated_qaly: float = 0.0
    detected: bool = False
    missed: bool = False
    screens: int = 0
    false_positives: int = 0

    def vector(self) -> NDArray[np.float32]:
        fibrosis = np.zeros(5, dtype=np.float32)
        fibrosis[int(self.fibrosis)] = 1.0
        trajectory = np.zeros(3, dtype=np.float32)
        trajectory[int(self.trajectory)] = 1.0
        return np.concatenate(
            (
                fibrosis,
                trajectory,
                np.asarray(
                    [
                        self.bmi / 50.0,
                        self.age / 100.0,
                        float(self.diabetes),
                        float(self.adequacy) / 2.0,
                        min(self.months_since_screen / 12.0, 2.0),
                        float(self.last_result),
                    ],
                    dtype=np.float32,
                ),
            )
        )


@dataclass(frozen=True)
class Transition:
    observation: NDArray[np.float32]
    action: int
    reward: float
    next_observation: NDArray[np.float32]
    terminated: bool
    constraints: NDArray[np.float32]
    next_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class EpisodeSummary:
    hcc_occurred: bool
    early_detected: bool
    any_detected: bool
    missed: bool
    screens: int
    cost: float
    qaly: float
    years: float
    false_positives: int
