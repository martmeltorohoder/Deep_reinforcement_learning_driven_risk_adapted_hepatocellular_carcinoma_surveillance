from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from scdsmdp.learning.trainer import Trainer
from scdsmdp.specification import load_spec


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="scdsmdp-train")
    value.add_argument("--config", type=Path, default=Path("settings/main.yaml"))
    value.add_argument("--output", type=Path, default=Path("runs/main"))
    value.add_argument("--resume", type=Path)
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trainer = Trainer(load_spec(arguments.config), arguments.output, torch.device(arguments.device))
    if arguments.resume is not None:
        trainer.resume(arguments.resume)
    trainer.train()


if __name__ == "__main__":
    main()
