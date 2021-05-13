import math
from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceNaNBasedOn(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        column_name: str,
        agg_func,
        group_by_column_names: Optional[List[str]] = None,
    ) -> None:
        self.column_name = column_name
        self.group_by_column_names = group_by_column_names
        self.agg_func = agg_func

        self.values_per_group = None

    def transform(self, df, **transform_params):
        def f(row):
            v = row[self.column_name]
            if v != v:
                if self.group_by_column_names is None:
                    return self.values_per_group

                if len(self.group_by_column_names) == 1:
                    key = row[self.group_by_column_names[0]]
                else:
                    key = tuple(row[k] for k in self.group_by_column_names)
                return self.values_per_group[key]

            return row[self.column_name]

        rows_with_nan = df[df[self.column_name].isna()]
        if len(rows_with_nan) == 0:
            return df

        replacement_values = rows_with_nan.apply(f, axis=1)
        df[self.column_name].fillna(replacement_values, inplace=True)

        return df

    def fit(self, df, y=None, **fit_params):
        if self.group_by_column_names is not None:
            self.values_per_group = (
                df.groupby(self.group_by_column_names).agg({self.column_name: self.agg_func}).to_dict()[self.column_name]
            )
        else:
            self.values_per_group = self.agg_func(df[self.column_name])
        return self


def most_common(x):
    return x.value_counts().index[0]
