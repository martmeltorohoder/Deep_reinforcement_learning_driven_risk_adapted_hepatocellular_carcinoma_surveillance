import numpy as np

from scdsmdp.environment.natural_history import NaturalHistory
from scdsmdp.environment.population import PopulationSampler
from scdsmdp.environment.screening import ScreeningModel
from scdsmdp.environment.simulator import SurveillanceSimulator
from scdsmdp.environment.types import Action, Adequacy, Fibrosis, Modality, Patient, Trajectory
from scdsmdp.specification import load_spec


def test_population_vectors_have_fourteen_features() -> None:
    patient = PopulationSampler().sample(np.random.default_rng(1))
    assert patient.vector().shape == (14,)
    assert patient.vector().dtype == np.float32


def test_action_round_trip() -> None:
    for index in range(18):
        assert Action.from_index(index).index == index


def test_severe_adequacy_masks_ultrasound() -> None:
    patient = Patient(Fibrosis.F3, Trajectory.STABLE, 38.0, 62.0, True, Adequacy.C)
    mask = ScreeningModel.action_mask(patient)
    assert mask.sum() == 12
    for interval in range(3):
        assert not mask[interval * 6 + int(Modality.US)]
        assert not mask[interval * 6 + int(Modality.US_AFP)]


def test_natural_history_is_seed_deterministic() -> None:
    first = Patient(Fibrosis.F2, Trajectory.PROGRESSING, 35.0, 60.0, True, Adequacy.B)
    second = Patient(Fibrosis.F2, Trajectory.PROGRESSING, 35.0, 60.0, True, Adequacy.B)
    history = NaturalHistory()
    first_rng = np.random.default_rng(91)
    second_rng = np.random.default_rng(91)
    for _ in range(20):
        history.advance(first, first_rng)
        history.advance(second, second_rng)
    assert first == second


def test_simulator_completes_horizon() -> None:
    spec = load_spec("settings/test.yaml")
    simulator = SurveillanceSimulator(spec)
    _, mask = simulator.reset()
    terminated = False
    steps = 0
    while not terminated:
        selected = int(np.flatnonzero(mask)[-1])
        result = simulator.step(selected)
        mask = result.mask
        terminated = result.terminated
        steps += 1
    assert steps <= 4
    assert simulator.summary().years <= 1.0
