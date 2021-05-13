import abc
from typing import Any, Dict, List, Set

import numpy as np
from sklearn.preprocessing import LabelEncoder


class PipelineLabelEncoder(LabelEncoder):
    def fit_transform(self, x, *args, **kwargs):
        return np.expand_dims(super().fit_transform(x), -1)

    def transform(self, x, *args, **kwargs):
        return np.expand_dims(super().transform(x), -1)
