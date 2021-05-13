from typing import Any, List, Optional, Sequence, Union

import torch


class EnsembleMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        n_inputs: int,
        n_hidden_units: Optional[Union[int, Sequence[int]]] = None,
        n_outputs: Optional[int] = None,
        n_heads: int = 10,
        dropout_prop: Optional[float] = None,
        batch_norm: bool = False,
        manual_expand_input: bool = False,
        manual_collapse_output: bool = False,
        activation: Any = None,
    ) -> None:
        super().__init__()

        if isinstance(n_hidden_units, int):
            n_hidden_units = [n_hidden_units]

        if n_hidden_units is None:
            n_hidden_units = []
        n_hidden_units = list(n_hidden_units)

        sequence: List[torch.nn.Module] = []
        for i, (n_in, n_out) in enumerate(zip([n_inputs] + list(n_hidden_units[:-1]), n_hidden_units)):
            layer = []
            layer.append(torch.nn.Conv1d(in_channels=n_in * n_heads, out_channels=n_out * n_heads, kernel_size=1, groups=n_heads))

            if activation is None:
                layer.append(torch.nn.ReLU())
            else:
                layer.append(activation())

            if batch_norm:
                layer.append(torch.nn.BatchNorm1d(n_out * n_heads))
            if dropout_prop is not None:
                layer.append(torch.nn.Dropout(dropout_prop))

            sequence.append(torch.nn.Sequential(*layer))

        if n_outputs is not None:
            last_size = n_hidden_units[-1] if n_hidden_units else n_inputs
            sequence.append(
                torch.nn.Conv1d(in_channels=last_size * n_heads, out_channels=n_outputs * n_heads, kernel_size=1, groups=n_heads)
            )

        self.model = torch.nn.Sequential(*sequence)
        self.n_outputs = n_hidden_units[-1] if n_outputs is None else n_outputs
        self.n_heads = n_heads
        self.manual_expand_input = manual_expand_input
        self.manual_collapse_output = manual_collapse_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if not self.manual_expand_input:
            x = self.expand(x.clone())

        x = self.model(x)

        if not self.manual_collapse_output:
            x = x.view(batch_size, self.n_heads, self.n_outputs)

        return x

    def expand(self, x: torch.Tensor):
        return x.repeat(1, self.n_heads).unsqueeze(-1)
