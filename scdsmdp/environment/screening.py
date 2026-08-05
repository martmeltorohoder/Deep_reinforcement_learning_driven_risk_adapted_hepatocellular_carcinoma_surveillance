from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scdsmdp.environment.types import Action, Adequacy, CancerStage, Modality, Patient


@dataclass(frozen=True)
class TestProfile:
    sensitivity: float
    specificity: float
    cost: float


class ScreeningModel:
    def __init__(self) -> None:
        self._fixed = {
            Modality.GALAD: TestProfile(0.811, 0.875, 280.0),
            Modality.AMRI: TestProfile(0.882, 0.891, 500.0),
            Modality.LIQUID_BIOPSY: TestProfile(0.74, 0.93, 450.0),
            Modality.NO_SCREEN: TestProfile(0.0, 1.0, 0.0),
        }

    def profile(self, modality: Modality, adequacy: Adequacy) -> TestProfile:
        if modality == Modality.US:
            sensitivity = (0.796, 0.47, 0.217)[int(adequacy)]
            return TestProfile(sensitivity, 0.91, 300.0)
        if modality == Modality.US_AFP:
            sensitivity = min((0.796, 0.47, 0.217)[int(adequacy)] + 0.12, 0.93)
            return TestProfile(sensitivity, 0.88, 350.0)
        return self._fixed[modality]

    def observe(
        self, patient: Patient, action: Action, rng: np.random.Generator
    ) -> tuple[bool, bool]:
        if action.modality == Modality.NO_SCREEN:
            patient.last_result = 0
            return False, False
        profile = self.profile(action.modality, patient.adequacy)
        patient.accumulated_cost += profile.cost
        patient.screens += 1
        patient.months_since_screen = 0
        if patient.cancer != CancerStage.NONE:
            positive = bool(rng.random() < profile.sensitivity)
            if positive:
                patient.detected = True
                patient.last_result = 1
                return True, False
            patient.last_result = 0
            return False, False
        false_positive = bool(rng.random() > profile.specificity)
        patient.last_result = int(false_positive)
        if false_positive:
            patient.false_positives += 1
            patient.accumulated_cost += 650.0
        return false_positive, false_positive

    @staticmethod
    def action_mask(patient: Patient) -> np.ndarray:
        mask = np.ones(18, dtype=np.bool_)
        if patient.adequacy == Adequacy.C:
            for interval in range(3):
                mask[interval * 6 + int(Modality.US)] = False
                mask[interval * 6 + int(Modality.US_AFP)] = False
        return mask
