from typing import Any, Union

import torch


def freeze(model: Union[torch.nn.Module, torch.nn.Parameter]) -> Union[torch.nn.Module, torch.nn.Parameter]:
    if isinstance(model, torch.nn.Module):
        for p in model.parameters():
            p.requires_grad = False
    elif isinstance(model, torch.nn.parameter.Parameter):
        model.requires_grad = False
    else:
        raise Exception(f"Unknown type: {type(model)}")

    return model


def clip_grad_if_need(parameters: Any, grad_norm_max: float) -> torch.Tensor:
    if grad_norm_max > 0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_norm_max)

    return None
