import numpy as np

from scdsmdp.assessment.metrics import aggregate, calibration_error
from scdsmdp.assessment.statistics import bootstrap_interval, interquartile_summary, paired_wilcoxon
from scdsmdp.environment.types import EpisodeSummary


def episode(detected: bool, early: bool, missed: bool, screens: int = 10) -> EpisodeSummary:
    return EpisodeSummary(True, early, detected, missed, screens, 1200.0, 6.5, 10.0, 1)


def test_metric_aggregation() -> None:
    metrics = aggregate([episode(True, True, False), episode(False, False, True)])
    assert metrics.early_stage_detection_rate == 0.5
    assert metrics.sensitivity == 0.5
    assert metrics.missed_cancer_rate == 0.5


def test_calibration_perfect_predictions() -> None:
    predicted = np.asarray([0.0, 0.0, 1.0, 1.0])
    observed = predicted.copy()
    expected, maximum, table = calibration_error(predicted, observed)
    assert expected == 0.0
    assert maximum == 0.0
    assert table.shape == (10, 4)


def test_bootstrap_contains_estimate() -> None:
    values = np.arange(20, dtype=np.float64)
    interval = bootstrap_interval(values, resamples=200, seed=3)
    assert interval.lower <= interval.estimate <= interval.upper


def test_wilcoxon_adjusts_p_value() -> None:
    treatment = np.arange(1, 21, dtype=np.float64)
    control = treatment - 2.0
    result = paired_wilcoxon(treatment, control)
    assert result.adjusted_p_value >= result.p_value
    assert result.effect_size > 0.0


def test_interquartile_summary() -> None:
    median, spread = interquartile_summary(np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert median == 2.5
    assert spread == 1.5
