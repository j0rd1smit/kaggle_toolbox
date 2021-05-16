from argparse import Namespace
from pathlib import Path
from typing import Dict, Union

import pytorch_lightning as pl
import yaml
from jsonargparse import namespace_to_dict
from pytorch_lightning import Callback


class LogConfig(Callback):
    def __init__(self, config: Union[Namespace, Dict], file_name: str = "config") -> None:
        self.config = config
        self.file_name = file_name

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log_config(trainer)

    def _log_config(self, trainer: "pl.Trainer"):
        if trainer.log_dir is None:
            return

        log_dir = Path(trainer.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        output_path = log_dir / f"{self.file_name}.yaml"

        with open(output_path, "w") as f:
            config = self.config
            if isinstance(config, Namespace):
                config = namespace_to_dict(self.config)

            yaml.dump(config, f)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._log_config(trainer)
