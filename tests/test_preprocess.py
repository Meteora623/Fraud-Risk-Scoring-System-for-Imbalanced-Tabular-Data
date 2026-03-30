from __future__ import annotations

import pandas as pd

from src.data.feature_pipeline import build_preprocessor
from src.data.preprocess import infer_feature_types
from src.data.schemas import DatasetSchema
from src.utils.config import PreprocessingConfig


def test_preprocessor_handles_paysim_numeric_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["CASH_OUT", None, "TRANSFER"],
            "amount": [1000.0, 2500.0, None],
            "oldbalanceOrg": [5000.0, 10000.0, 0.0],
            "newbalanceOrig": [4000.0, 7500.0, 0.0],
            "oldbalanceDest": [0.0, 500.0, 200.0],
            "newbalanceDest": [1000.0, 3000.0, 1200.0],
            "nameOrig": ["C123", "C234", "C345"],
            "nameDest": ["M111", "M222", "M333"],
            "isFraud": [0, 1, 0],
        }
    )
    schema = DatasetSchema(target_col="isFraud", id_cols=["nameOrig", "nameDest"], time_col="step")

    features = infer_feature_types(df, schema)
    preprocessor = build_preprocessor(features, PreprocessingConfig(), for_linear_model=True)

    x = df.drop(columns=["isFraud"])
    transformed = preprocessor.fit_transform(x)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 6
    assert "type" in features.categorical_cols
