import abc
from typing import Any, Callable, Dict, List, Set

import torch


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)
