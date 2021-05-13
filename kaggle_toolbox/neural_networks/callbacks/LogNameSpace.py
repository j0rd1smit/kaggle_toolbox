from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import yaml
from jsonargparse import namespace_to_dict
from pytorch_lightning import Callback


class LogNameSpace(Callback):
    def __init__(self, name_space: Namespace, file_name: str = "config") -> None:
        self.name_space = name_space
        self.file_name = file_name

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.log_dir is None:
            return

        log_dir = Path(trainer.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        output_path = log_dir / f"{self.file_name}.yaml"

        with open(output_path, "w") as f:
            yaml.dump(namespace_to_dict(self.name_space), f)
