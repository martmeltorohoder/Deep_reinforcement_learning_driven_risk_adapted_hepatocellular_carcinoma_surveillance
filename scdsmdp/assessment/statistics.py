from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int


@dataclass(frozen=True)
class PairedTest:
    statistic: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    significant: bool


def bootstrap_interval(
    values: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = statistic(sample)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, (tail, 1.0 - tail))
    return Interval(float(statistic(values)), float(lower), float(upper), confidence, resamples)


def paired_wilcoxon(
    treatment: NDArray[np.float64],
    control: NDArray[np.float64],
    comparisons: int = 15,
    alpha: float = 0.05,
) -> PairedTest:
    if treatment.shape != control.shape or treatment.ndim != 1:
        raise ValueError("paired vectors must have identical shapes")
    result = stats.wilcoxon(treatment, control, alternative="two-sided")
    differences = treatment - control
    scale = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    effect = (
        float(np.median(differences) / scale)
        if scale > 0.0
        else float(np.sign(np.median(differences)) * np.inf)
    )
    adjusted = min(float(result.pvalue) * comparisons, 1.0)
    return PairedTest(
        float(result.statistic), float(result.pvalue), adjusted, effect, adjusted < alpha
    )


def interquartile_summary(values: NDArray[np.float64]) -> tuple[float, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty vector")
    first, median, third = np.quantile(values, (0.25, 0.5, 0.75))
    return float(median), float(third - first)


def weighted_importance_sampling(
    rewards: NDArray[np.float64],
    target_probabilities: NDArray[np.float64],
    behavior_probabilities: NDArray[np.float64],
    episode_ids: NDArray[np.int64],
    discount: float = 0.95,
) -> float:
    if not (
        rewards.shape
        == target_probabilities.shape
        == behavior_probabilities.shape
        == episode_ids.shape
    ):
        raise ValueError("all inputs must align")
    estimates: list[float] = []
    weights: list[float] = []
    for episode in np.unique(episode_ids):
        selected = episode_ids == episode
        ratios = target_probabilities[selected] / np.clip(
            behavior_probabilities[selected], 1e-12, None
        )
        cumulative = np.cumprod(ratios)
        powers = discount ** np.arange(selected.sum())
        estimates.append(float(np.sum(powers * cumulative * rewards[selected])))
        weights.append(float(cumulative[-1]))
    normalizer = sum(weights)
    if normalizer <= 0.0:
        return 0.0
    return float(np.dot(estimates, weights) / normalizer)
