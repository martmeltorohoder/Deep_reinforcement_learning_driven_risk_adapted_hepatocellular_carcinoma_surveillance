from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scdsmdp.assessment.evaluator import Evaluator
from scdsmdp.learning.baselines import registry
from scdsmdp.specification import load_spec


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="scdsmdp-evaluate")
    value.add_argument("--config", type=Path, default=Path("settings/main.yaml"))
    value.add_argument("--policy", choices=tuple(registry()), default="risk_adaptive")
    value.add_argument("--episodes", type=int, default=50000)
    value.add_argument("--seed", type=int, default=1000)
    return value


def main() -> None:
    arguments = parser().parse_args()
    spec = load_spec(arguments.config)
    policy = registry(arguments.seed)[arguments.policy]
    evaluator = Evaluator(spec, arguments.seed)
    metrics = evaluator.evaluate(
        lambda observation, mask, patient: policy.select(patient, mask), arguments.episodes
    )
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
