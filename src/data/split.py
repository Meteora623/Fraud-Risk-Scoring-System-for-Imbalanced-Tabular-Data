from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schemas import DatasetSchema
from src.utils.config import SplitConfig


def split_dataset(
    df: pd.DataFrame,
    schema: DatasetSchema,
    split_cfg: SplitConfig,
    random_seed: int,
) -> Dict[str, pd.DataFrame]:
    target = df[schema.target_col]

    if split_cfg.strategy == "time_based" and schema.time_col:
        sorted_df = df.sort_values(schema.time_col).reset_index(drop=True)
        n = len(sorted_df)
        train_end = int(n * split_cfg.train_size)
        val_end = train_end + int(n * split_cfg.val_size)
        return {
            "train": sorted_df.iloc[:train_end].copy(),
            "val": sorted_df.iloc[train_end:val_end].copy(),
            "test": sorted_df.iloc[val_end:].copy(),
        }

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - split_cfg.train_size),
        random_state=random_seed,
        stratify=target,
    )

    relative_test = split_cfg.test_size / (split_cfg.val_size + split_cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_seed,
        stratify=temp_df[schema.target_col],
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
