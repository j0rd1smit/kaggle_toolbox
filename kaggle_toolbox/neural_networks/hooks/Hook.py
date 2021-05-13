import numpy as np
import torch


def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if isinstance(x, list):
        return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x, dict):
        return {k: apply(func, v, *args, **kwargs) for k, v in x.items()}
    res = func(x, *args, **kwargs)
    return res


def to_detach(b, cpu=True, gather=True):
    "Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`."

    def _inner(x, cpu=True, gather=True):
        if not isinstance(x, torch.Tensor):
            return x
        x = x.detach()

        return x.cpu() if cpu else x

    return apply(_inner, b, cpu=cpu, gather=gather)


class Hook:
    def __init__(self, module, hook_func, detach=True, cpu=False, gather=False):
        self.hook_func = hook_func
        self.detach = detach
        self.cpu = cpu
        self.gather = gather

        self.hook = module.register_forward_hook(self.hook_fn)
        self._stored = None
        self.removed = False

    @property
    def stored(self):
        assert self._stored is not None
        return self._stored

    def hook_fn(self, module, inputs, outputs):
        if self.detach:
            inputs = to_detach(inputs, cpu=self.cpu, gather=self.gather)
            outputs = to_detach(outputs, cpu=self.cpu, gather=self.gather)

        self._stored = self.hook_func(module, inputs, outputs)

    def clear(self) -> None:
        self._stored = None

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()
