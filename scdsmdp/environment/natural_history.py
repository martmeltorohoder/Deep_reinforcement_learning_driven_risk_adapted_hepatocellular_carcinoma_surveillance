from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scdsmdp.environment.types import CancerStage, Fibrosis, Patient, Trajectory


@dataclass(frozen=True)
class NaturalHistoryParameters:
    years_per_stage: tuple[float, float, float, float] = (9.9, 10.3, 13.3, 22.2)
    regression_annual: tuple[float, float, float, float, float] = (0.0, 0.025, 0.022, 0.018, 0.01)
    noncirrhotic_hcc_per_year: float = 0.00008
    cirrhotic_hcc_per_year: float = 0.0106
    diabetes_hazard: float = 2.27
    progressing_hazard: float = 2.31
    obesity_hazard: float = 1.35
    decompensation_annual: float = 0.04
    background_death_annual: float = 0.012
    early_to_advanced_quarterly: float = 0.29


class NaturalHistory:
    def __init__(self, parameters: NaturalHistoryParameters | None = None) -> None:
        self.parameters = parameters or NaturalHistoryParameters()

    def _annual_progression(self, patient: Patient) -> float:
        if patient.fibrosis == Fibrosis.F4:
            return self.parameters.decompensation_annual
        base = 1.0 / self.parameters.years_per_stage[int(patient.fibrosis)]
        multiplier = 1.0
        if patient.diabetes:
            multiplier *= self.parameters.diabetes_hazard
        if patient.trajectory == Trajectory.PROGRESSING:
            multiplier *= self.parameters.progressing_hazard
        if patient.trajectory == Trajectory.REGRESSING:
            multiplier *= 0.55
        if patient.bmi >= 30.0:
            multiplier *= self.parameters.obesity_hazard
        return min(base * multiplier, 0.95)

    def _annual_hcc(self, patient: Patient) -> float:
        if patient.fibrosis == Fibrosis.F4:
            base = self.parameters.cirrhotic_hcc_per_year
        else:
            stage_scale = (0.35, 0.5, 0.8, 1.6)[int(patient.fibrosis)]
            base = self.parameters.noncirrhotic_hcc_per_year * stage_scale
        if patient.diabetes:
            base *= self.parameters.diabetes_hazard
        if patient.trajectory == Trajectory.PROGRESSING:
            base *= self.parameters.progressing_hazard
        return min(base, 0.5)

    @staticmethod
    def _quarterly(annual: float) -> float:
        return float(1.0 - (1.0 - annual) ** 0.25)

    def advance(self, patient: Patient, rng: np.random.Generator) -> None:
        if not patient.alive:
            return
        if rng.random() < self._quarterly(self.parameters.background_death_annual):
            patient.alive = False
            return
        if patient.cancer == CancerStage.EARLY:
            if rng.random() < self.parameters.early_to_advanced_quarterly:
                patient.cancer = CancerStage.ADVANCED
        elif patient.cancer == CancerStage.ADVANCED:
            if rng.random() < 0.12:
                patient.alive = False
                return
        elif rng.random() < self._quarterly(self._annual_hcc(patient)):
            patient.cancer = CancerStage.EARLY
        progression = self._quarterly(self._annual_progression(patient))
        regression = self._quarterly(self.parameters.regression_annual[int(patient.fibrosis)])
        draw = rng.random()
        if draw < regression and patient.fibrosis > Fibrosis.F0:
            patient.fibrosis = Fibrosis(int(patient.fibrosis) - 1)
        elif draw < regression + progression:
            if patient.fibrosis < Fibrosis.F4:
                patient.fibrosis = Fibrosis(int(patient.fibrosis) + 1)
            else:
                patient.decompensated = True
        self._update_trajectory(patient, rng)
        patient.age += 0.25
        patient.months_since_screen += 3

    @staticmethod
    def _update_trajectory(patient: Patient, rng: np.random.Generator) -> None:
        transition = np.asarray(
            (
                (0.82, 0.16, 0.02),
                (0.08, 0.86, 0.06),
                (0.03, 0.18, 0.79),
            ),
            dtype=np.float64,
        )
        patient.trajectory = Trajectory(rng.choice(3, p=transition[int(patient.trajectory)]))
