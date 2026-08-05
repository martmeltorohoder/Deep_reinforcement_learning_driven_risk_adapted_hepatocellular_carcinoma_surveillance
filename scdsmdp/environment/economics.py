from __future__ import annotations

from scdsmdp.environment.types import CancerStage, Patient


class HealthEconomics:
    def __init__(self, annual_discount: float = 0.03) -> None:
        self.annual_discount = annual_discount

    def quarterly_qaly(self, patient: Patient, quarter: int) -> float:
        if not patient.alive:
            utility = 0.0
        elif patient.cancer == CancerStage.EARLY:
            utility = 0.76
        elif patient.cancer == CancerStage.ADVANCED:
            utility = 0.44
        elif patient.decompensated:
            utility = 0.60
        else:
            utility = 0.84
        factor = (1.0 + self.annual_discount) ** (-(quarter / 4.0))
        value = float(utility * 0.25 * factor)
        patient.accumulated_qaly += value
        return value

    def discounted_cost(self, amount: float, quarter: int) -> float:
        return float(amount * (1.0 + self.annual_discount) ** (-(quarter / 4.0)))

    @staticmethod
    def treatment_cost(patient: Patient) -> float:
        if not patient.detected:
            return 0.0
        if patient.cancer == CancerStage.EARLY:
            return 42000.0
        if patient.cancer == CancerStage.ADVANCED:
            return 118000.0
        return 0.0
