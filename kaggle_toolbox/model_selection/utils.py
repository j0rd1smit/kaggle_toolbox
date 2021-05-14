from typing import Optional, Sequence, TypeVar

import numpy as np
from sklearn.model_selection import train_test_split

T = TypeVar("T")


def train_val_test_split(
    *data: Sequence[T],
    val_size: float,
    test_size: float,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: np.ndarray = None,
) -> Sequence[T]:
    n_inputs = len(data)
    other_size = val_size + test_size

    train_and_other_data = train_test_split(
        *data,
        test_size=other_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    train_data = train_and_other_data[0::2]
    other_data = train_and_other_data[1::2]

    if stratify is not None:
        train_and_other_stratify = train_test_split(
            stratify,
            test_size=other_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        stratify = train_and_other_stratify[1]

    val_and_test_data = train_test_split(
        *other_data,
        test_size=test_size / other_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    val_data = val_and_test_data[0::2]
    test_data = val_and_test_data[1::2]

    result = []
    for i in range(n_inputs):
        result.append(train_data[i])
        result.append(val_data[i])
        result.append(test_data[i])

    return result
