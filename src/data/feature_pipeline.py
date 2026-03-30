from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.preprocess import FeatureSet
from src.utils.config import PreprocessingConfig


def _replace_none_with_nan(x):
    frame = pd.DataFrame(x).replace({None: np.nan})
    return frame


def build_preprocessor(feature_set: FeatureSet, cfg: PreprocessingConfig, for_linear_model: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy=cfg.numeric_imputer_strategy))]
    if for_linear_model and cfg.scale_numeric_for_linear:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=cfg.one_hot_min_frequency,
    )

    categorical_steps = [
        (
            "normalize_nulls",
            FunctionTransformer(_replace_none_with_nan, feature_names_out="one-to-one"),
        ),
        ("imputer", SimpleImputer(strategy=cfg.categorical_imputer_strategy)),
        ("encoder", categorical_encoder),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), feature_set.numeric_cols),
            ("cat", Pipeline(categorical_steps), feature_set.categorical_cols),
        ],
        remainder="drop",
    )
