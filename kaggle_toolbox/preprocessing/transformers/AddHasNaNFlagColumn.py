from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AddHasNaNFlagColumn(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        columns: Optional[List[str]] = None,
        present_suffix: Optional[str] = None,
    ) -> None:
        self._specified_columns = columns
        self._columns = columns if columns is not None else None
        self.present_suffix = present_suffix if present_suffix is not None else "_present"

    def transform(self, df, **transform_params):
        assert self._columns is not None
        assert len(self._columns) > 0
        df = pd.concat([df, df[self._columns].notnull().astype(int).add_suffix(self.present_suffix)], axis=1)

        return df

    def fit(self, df, y=None, **fit_params):
        if self._columns is None:
            nans_per_column = df.isna().sum(0)
            self._columns = nans_per_column[nans_per_column > 0].index

        return self
