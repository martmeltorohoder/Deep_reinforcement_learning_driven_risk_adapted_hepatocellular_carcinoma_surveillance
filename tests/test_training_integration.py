from pathlib import Path

import torch

from scdsmdp.learning.trainer import Trainer
from scdsmdp.specification import load_spec


def test_two_episode_training_and_resume(tmp_path: Path) -> None:
    spec = load_spec("settings/test.yaml")
    output = tmp_path / "run"
    trainer = Trainer(spec, output, torch.device("cpu"))
    reports = trainer.train()
    assert len(reports) == 2
    assert (output / "final.pt").exists()
    resumed = Trainer(spec, tmp_path / "resumed", torch.device("cpu"))
    resumed.resume(output / "final.pt")
    assert resumed.start_episode == 2
    assert resumed.steps > 0
