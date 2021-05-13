from typing import Any, Callable, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MapColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        column_name: str,
        func: Callable[[Any], Any],
        *,
        target_column: str = None,
    ) -> None:
        self.column_name = column_name
        self.func = func
        self.target_column = target_column if target_column is not None else column_name

    def transform(self, df: pd.DataFrame, **transform_params):
        df[self.target_column] = df[self.column_name].map(self.func)
        return df

    def fit(self, X, y=None, **fit_params):
        return self
