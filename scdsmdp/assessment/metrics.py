from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scdsmdp.environment.types import EpisodeSummary


@dataclass(frozen=True)
class SurveillanceMetrics:
    early_stage_detection_rate: float
    sensitivity: float
    missed_cancer_rate: float
    overscreening_reduction: float
    incremental_cost_effectiveness: float
    screens_per_year: float
    mean_qaly: float
    mean_cost: float
    false_positive_rate: float
    episodes: int
    hcc_events: int


def aggregate(
    episodes: list[EpisodeSummary],
    reference_screens_per_year: float = 2.0,
    no_surveillance_qaly: float = 5.7,
    no_surveillance_cost: float = 0.0,
) -> SurveillanceMetrics:
    if not episodes:
        raise ValueError("at least one episode is required")
    hcc = [episode for episode in episodes if episode.hcc_occurred]
    early = sum(episode.early_detected for episode in hcc)
    detected = sum(episode.any_detected for episode in hcc)
    missed = sum(episode.missed for episode in hcc)
    screens = sum(episode.screens for episode in episodes)
    years = sum(episode.years for episode in episodes)
    mean_screens = screens / max(years, 1e-12)
    overscreening = 1.0 - mean_screens / reference_screens_per_year
    mean_qaly = float(np.mean([episode.qaly for episode in episodes]))
    mean_cost = float(np.mean([episode.cost for episode in episodes]))
    qaly_gain = mean_qaly - no_surveillance_qaly
    icer = (mean_cost - no_surveillance_cost) / qaly_gain if qaly_gain > 0.0 else float("inf")
    false_positives = sum(episode.false_positives for episode in episodes)
    return SurveillanceMetrics(
        early / max(len(hcc), 1),
        detected / max(len(hcc), 1),
        missed / max(len(hcc), 1),
        overscreening,
        icer,
        mean_screens,
        mean_qaly,
        mean_cost,
        false_positives / max(screens, 1),
        len(episodes),
        len(hcc),
    )


def calibration_error(
    predicted: np.ndarray,
    observed: np.ndarray,
    bins: int = 10,
) -> tuple[float, float, np.ndarray]:
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed shapes differ")
    if predicted.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(predicted, edges[1:-1]), bins - 1)
    table = np.full((bins, 4), np.nan, dtype=np.float64)
    weighted = 0.0
    maximum = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        table[index, 0] = edges[index]
        table[index, 1] = edges[index + 1]
        table[index, 3] = count
        if count:
            mean_prediction = float(predicted[selected].mean())
            mean_observed = float(observed[selected].mean())
            difference = abs(mean_prediction - mean_observed)
            table[index, 2] = difference
            weighted += count * difference
            maximum = max(maximum, difference)
    return weighted / max(len(predicted), 1), maximum, table


def subgroup_metrics(
    episodes: list[EpisodeSummary],
    labels: list[str],
) -> dict[str, SurveillanceMetrics]:
    if len(episodes) != len(labels):
        raise ValueError("episodes and labels must align")
    result: dict[str, SurveillanceMetrics] = {}
    for label in sorted(set(labels)):
        selected = [
            episode
            for episode, candidate in zip(episodes, labels, strict=True)
            if candidate == label
        ]
        result[label] = aggregate(selected)
    return result


def disparity(values: dict[str, SurveillanceMetrics], field: str) -> float:
    scores = [float(getattr(metric, field)) for metric in values.values()]
    return max(scores) - min(scores) if scores else 0.0
