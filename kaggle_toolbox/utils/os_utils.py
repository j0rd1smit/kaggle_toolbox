import os
from pathlib import Path
from typing import Union

StrOrPath = Union[str, Path]


def relative_to(file: StrOrPath, relative: str) -> Path:
    dir_name = os.path.dirname(file)

    return Path(os.path.join(dir_name, relative))
