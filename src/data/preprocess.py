from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.data.schemas import DatasetSchema


@dataclass
class FeatureSet:
    numeric_cols: List[str]
    categorical_cols: List[str]


def infer_feature_types(df: pd.DataFrame, schema: DatasetSchema) -> FeatureSet:
    drop_cols = set(schema.id_cols + [schema.target_col])
    features = [c for c in df.columns if c not in drop_cols]
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]
    return FeatureSet(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
