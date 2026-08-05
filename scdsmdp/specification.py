from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RewardSpec:
    detection: float = 10.0
    qaly: float = 1.0
    cost: float = 0.01
    burden: float = 5.0


@dataclass(frozen=True)
class ConstraintSpec:
    missed_cancer: float = 0.01
    interval_violation: float = 0.0
    annual_budget: float = 3000.0
    dual_learning_rate: float = 0.001
    update_frequency: int = 100
    initial_multiplier: float = 1.0


@dataclass(frozen=True)
class ModelSpec:
    state_size: int = 14
    action_size: int = 18
    hidden_size: int = 256
    hidden_layers: int = 3
    quantile_basis: int = 64
    quantile_samples: int = 64
    cvar_level: float = 0.05


@dataclass(frozen=True)
class TrainingSpec:
    episodes: int = 680000
    batch_size: int = 256
    learning_rate: float = 0.0003
    discount: float = 0.95
    cql_weight: float = 1.0
    target_rate: float = 0.005
    replay_capacity: int = 2000000
    learning_starts: int = 10000
    train_frequency: int = 4
    gradient_clip: float = 10.0
    evaluation_frequency: int = 10000
    evaluation_episodes: int = 50000
    seeds: tuple[int, ...] = tuple(range(20))


@dataclass(frozen=True)
class SimulatorSpec:
    horizon_years: int = 10
    decisions_per_year: int = 4
    economic_discount: float = 0.03
    population_size: int = 1000000


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int = 0
    reward: RewardSpec = field(default_factory=RewardSpec)
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    simulator: SimulatorSpec = field(default_factory=SimulatorSpec)


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    raw = values.get(name, {})
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping")
    return raw


def load_spec(path: str | Path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")
    training = _section(raw, "training")
    if "seeds" in training:
        training["seeds"] = tuple(int(value) for value in training["seeds"])
    return ExperimentSpec(
        seed=int(raw.get("seed", 0)),
        reward=RewardSpec(**_section(raw, "reward")),
        constraints=ConstraintSpec(**_section(raw, "constraints")),
        model=ModelSpec(**_section(raw, "model")),
        training=TrainingSpec(**training),
        simulator=SimulatorSpec(**_section(raw, "simulator")),
    )
