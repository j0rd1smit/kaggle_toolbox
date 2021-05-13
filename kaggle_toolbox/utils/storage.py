import datetime
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from kaggle_toolbox.utils.os_utils import StrOrPath


def store_model(
    root_dir: StrOrPath,
    model: Any,
    *,
    hparams: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    exist_ok: bool = False,
) -> Path:
    storage_path = timestamped_log_path(root_dir, prefix=prefix)
    storage_path.mkdir(parents=True, exist_ok=exist_ok)

    with open(storage_path / "model", "wb") as f:
        f.write(pickle.dumps(model))

    if hparams is not None:
        with open(storage_path / "hparams.yml", "w") as f:
            yaml.dump(hparams, f)

    if metrics is not None:
        # TODO add suport for df

        for k, v in metrics.items():
            if type(v).__module__ == np.__name__:
                metrics[k] = v.item()

        print(metrics)

        with open(storage_path / "metrics.yml", "w") as f:
            yaml.dump(metrics, f)

    return storage_path


def timestamped_log_path(
    root_dir: StrOrPath,
    *,
    prefix: str = "",
) -> Path:
    root_dir = Path(root_dir)

    if len(prefix) > 0:
        root_dir = root_dir / prefix

    now = datetime.datetime.now()
    dir_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

    return root_dir / dir_name
