from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schemas import DatasetSchema
from src.utils.config import ProjectConfig


LEGACY_TARGET_ALIASES = ["Class", "class", "is_fraud", "fraud", "target", "label"]

PAYSIM_COLUMN_ALIASES = {
    "step": ["step", "time_step", "time"],
    "type": ["type", "transaction_type"],
    "amount": ["amount", "transaction_amount"],
    "oldbalanceOrg": ["oldbalanceOrg", "oldbalanceorig", "old_balance_org"],
    "newbalanceOrig": ["newbalanceOrig", "newbalanceorg", "new_balance_orig"],
    "oldbalanceDest": ["oldbalanceDest", "oldbalancedest", "old_balance_dest"],
    "newbalanceDest": ["newbalanceDest", "newbalancedest", "new_balance_dest"],
    "nameOrig": ["nameOrig", "source_account", "source_id"],
    "nameDest": ["nameDest", "destination_account", "destination_id"],
    "isFraud": ["isFraud", "fraud", "target", "label", "class"],
    "isFlaggedFraud": ["isFlaggedFraud", "flagged_fraud", "is_flagged_fraud"],
}


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _rename_with_aliases(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    lower_to_actual = {c.lower(): c for c in df.columns}
    rename_map: dict[str, str] = {}

    for canonical, candidates in alias_map.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            actual = lower_to_actual.get(candidate.lower())
            if actual and actual not in rename_map:
                rename_map[actual] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _coerce_target(df: pd.DataFrame, target_col: str, positive_value: int) -> pd.DataFrame:
    if df[target_col].dtype == object:
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "fraud", "yes", "y"])
            .astype(int)
        )
    else:
        df[target_col] = (df[target_col] == positive_value).astype(int)
    return df


def _resolve_target_column(df: pd.DataFrame, target_col: str, aliases: list[str]) -> str:
    if target_col in df.columns:
        return target_col

    lower_to_actual = {c.lower(): c for c in df.columns}
    for alias in aliases:
        found = lower_to_actual.get(alias.lower())
        if found:
            return found

    raise ValueError(f"Target column '{target_col}' not found and no alias target column exists.")


def _sample_if_needed(df: pd.DataFrame, target_col: str, max_rows: int | None, random_seed: int) -> tuple[pd.DataFrame, bool]:
    if max_rows is None or len(df) <= max_rows:
        return df, False

    sampled = df.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, int(round(max_rows * (len(x) / len(df))))),
            random_state=random_seed,
        )
    )
    sampled = sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)

    return sampled, True


def normalize_schema(df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, DatasetSchema]:
    df = df.copy()
    raw_row_count = len(df)

    if config.dataset.name.lower() == "paysim":
        df = _rename_with_aliases(df, PAYSIM_COLUMN_ALIASES)

    target_col = config.schema.target_col
    candidate_aliases = list(config.dataset.target_aliases) + LEGACY_TARGET_ALIASES
    resolved_target = _resolve_target_column(df, target_col, candidate_aliases)
    if resolved_target != target_col:
        df = df.rename(columns={resolved_target: target_col})

    df = _coerce_target(df, target_col, config.schema.positive_class_value)

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper()

    for numeric_col in [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    dropped_identifier_cols = [c for c in config.dataset.drop_identifier_columns if c in df.columns]
    if dropped_identifier_cols:
        df = df.drop(columns=dropped_identifier_cols)

    dropped_leakage_cols = [
        c for c in config.dataset.leakage_risk_columns if c in df.columns and c != target_col
    ]
    if dropped_leakage_cols:
        df = df.drop(columns=dropped_leakage_cols)

    duplicates_removed = 0
    if config.dataset.duplicate_policy == "drop_exact":
        duplicates_before = int(df.duplicated().sum())
        if duplicates_before:
            df = df.drop_duplicates().reset_index(drop=True)
            duplicates_removed = duplicates_before

    df, sampling_applied = _sample_if_needed(
        df,
        target_col=target_col,
        max_rows=config.dataset.sample_max_rows,
        random_seed=config.random_seed,
    )

    schema = DatasetSchema(
        target_col=target_col,
        id_cols=[c for c in config.schema.id_cols if c in df.columns],
        time_col=config.schema.time_col if config.schema.time_col in df.columns else None,
        dataset_name=config.dataset.name,
        dataset_source=config.dataset.source,
        provenance_url=config.dataset.provenance_url,
        is_synthetic=config.dataset.is_synthetic,
        dropped_identifier_cols=dropped_identifier_cols,
        dropped_leakage_cols=dropped_leakage_cols,
        duplicate_policy=config.dataset.duplicate_policy,
        duplicates_removed=duplicates_removed,
        raw_row_count=raw_row_count,
        sampling_applied=sampling_applied,
        sample_max_rows=config.dataset.sample_max_rows,
        sampled_row_count=len(df),
    )
    return df, schema


def load_and_normalize(config: ProjectConfig) -> tuple[pd.DataFrame, DatasetSchema]:
    df = load_table(config.paths.raw_data_path)
    return normalize_schema(df, config)
