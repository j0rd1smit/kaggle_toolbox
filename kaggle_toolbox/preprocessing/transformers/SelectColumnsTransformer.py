from typing import List

from sklearn.base import BaseEstimator, TransformerMixin


class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self
