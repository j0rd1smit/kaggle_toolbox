from kaggle_toolbox.neural_networks.hooks.Hook import Hook


class Hooks:
    def __init__(self, modules, hook_func, detach=True, cpu=False, gather=False):
        self.hooks = [Hook(m, hook_func, detach=detach, cpu=cpu, gather=gather) for m in modules]

    def __getitem__(self, i):
        return self.hooks[i]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return list(o.stored for o in self)

    def remove(self):
        for h in self.hooks:
            h.remove()

    def clear(self) -> None:
        for hook in self.hooks:
            hook.clear()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()
