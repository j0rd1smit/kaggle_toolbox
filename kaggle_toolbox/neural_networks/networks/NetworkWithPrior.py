from typing import Optional

import torch

from kaggle_toolbox.neural_networks.utils import freeze


class NetworkWithPrior(torch.nn.Module):
    def __init__(
        self,
        network: torch.nn.Module,
        prior_network: torch.nn.Module,
        prior_scale: float,
        shared_encoder: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder

        self.network = network
        self.prior_network = freeze(prior_network)
        self.prior_scale = prior_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_encoder is not None:
            x = self.shared_encoder(x)

        return self.network(x) + self.prior(x)

    def prior(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.prior_scale * self.prior_network(x)
