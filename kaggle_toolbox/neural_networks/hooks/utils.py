from typing import List, Union

import torch

from kaggle_toolbox.neural_networks.hooks.Hooks import Hooks


def hook_outputs(modules, detach: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
    def _hook_inner(m, i, o):
        if isinstance(o, torch.Tensor):
            return o
        return list(o)

    return Hooks(modules, _hook_inner, detach=detach)


def hook_mlp_hidden_units(modules: torch.nn.Sequential, detach: bool = True) -> Hooks:
    modules = [modules[i] for i in range(0, len(modules) - 1)]

    return hook_outputs(modules, detach=detach)
