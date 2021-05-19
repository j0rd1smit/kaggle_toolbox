from typing import List, Optional, Sequence, Union

import numpy as np
import torch.nn


def mlp(
    *,
    n_inputs: int,
    n_hidden_units: Optional[Union[int, Sequence[int]]] = None,
    n_outputs: Optional[int] = None,
    dropout_probs: Optional[Union[float, Sequence[float]]] = None,
    batch_norm: bool = False,
    bn_1st: bool = True,
    activation: Optional[torch.nn.Module] = None,
) -> torch.nn.Module:
    if isinstance(n_hidden_units, int):
        n_hidden_units = [n_hidden_units]

    if n_hidden_units is None:
        n_hidden_units = []
    n_hidden_units = list(n_hidden_units)

    if isinstance(dropout_probs, (np.int_, np.float_)):
        dropout_probs = dropout_probs.item()

    if isinstance(dropout_probs, (float, int)):
        dropout_probs = [dropout_probs for _ in n_hidden_units]
    if dropout_probs is None:
        dropout_probs = [0 for _ in n_hidden_units]

    dropout_probs = list(dropout_probs)

    assert len(dropout_probs) == len(
        n_hidden_units
    ), f"Dropout probs must have same length as hidden_sizes but {len(dropout_probs)} != {len(n_hidden_units)}"
    assert all(
        (0 <= dropout_prob < 1 for dropout_prob in dropout_probs)
    ), f"Dropout prob must be in range [0, 1] but probs are {dropout_probs}"

    sequence: List[torch.nn.Module] = []

    for i, (n_in, n_out, dropout_prob) in enumerate(zip([n_inputs] + list(n_hidden_units[:-1]), n_hidden_units, dropout_probs)):
        layer = []
        layer.append(torch.nn.Linear(n_in, n_out))

        if batch_norm and bn_1st:
            layer.append(torch.nn.BatchNorm1d(n_out))

        if activation is None:
            layer.append(torch.nn.ReLU())
        else:
            layer.append(activation())

        if batch_norm and not bn_1st:
            layer.append(torch.nn.BatchNorm1d(n_out))

        if dropout_prob > 0:
            layer.append(torch.nn.Dropout(p=dropout_prob))

        sequence.append(torch.nn.Sequential(*layer))

    if n_outputs is not None:
        last_size = n_hidden_units[-1] if n_hidden_units else n_inputs
        sequence.append(torch.nn.Linear(last_size, n_outputs))

    return torch.nn.Sequential(*sequence)


if __name__ == "__main__":
    m = mlp(n_inputs=1, n_hidden_units=[8, 8], n_outputs=1, dropout_probs=np.array([0.1, 0.2]))
    print(m)
