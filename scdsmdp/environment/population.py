from __future__ import annotations

import numpy as np

from scdsmdp.environment.types import Adequacy, Fibrosis, Patient, Trajectory


class PopulationSampler:
    def sample(self, rng: np.random.Generator) -> Patient:
        fibrosis = Fibrosis(rng.choice(5, p=(0.40, 0.282, 0.105, 0.082, 0.131)))
        trajectory = Trajectory(rng.choice(3, p=(0.19, 0.70, 0.11)))
        age = float(np.clip(rng.normal(56.0, 13.5), 18.0, 90.0))
        diabetes = bool(rng.random() < 0.29)
        bmi = float(np.clip(rng.normal(30.2 if diabetes else 28.1, 5.8), 16.0, 55.0))
        adequacy = self._adequacy(bmi, rng)
        return Patient(fibrosis, trajectory, bmi, age, diabetes, adequacy)

    @staticmethod
    def _adequacy(bmi: float, rng: np.random.Generator) -> Adequacy:
        severe = 0.08
        moderate = 0.20
        if bmi >= 30.0:
            severe *= 2.08
            moderate *= 1.35
        if bmi >= 35.0:
            severe *= 1.45
        draw = rng.random()
        if draw < min(severe, 0.55):
            return Adequacy.C
        if draw < min(severe + moderate, 0.85):
            return Adequacy.B
        return Adequacy.A
