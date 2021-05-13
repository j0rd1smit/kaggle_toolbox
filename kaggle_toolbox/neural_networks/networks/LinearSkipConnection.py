import torch


class LinearSkipConnection(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        n_inputs: int,
        n_outputs: int,
    ) -> None:
        super().__init__()
        self.network = network

        self.linear_skip_connection = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.network(x) + 0.5 * self.linear_skip_connection(x)
