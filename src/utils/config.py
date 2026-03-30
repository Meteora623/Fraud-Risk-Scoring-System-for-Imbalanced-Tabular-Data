from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PathsConfig(BaseModel):
    raw_data_path: str
    processed_data_path: str
    split_data_dir: str
    audit_report_path: str
    model_dir: str
    artifact_dir: str
    comparison_metrics_path: str
    threshold_summary_path: str


class DatasetConfig(BaseModel):
    name: str = "paysim"
    source: str = "kaggle"
    provenance_url: str | None = None
    kaggle_dataset: str | None = "ealaxi/paysim1"
    is_synthetic: bool = True
    target_aliases: List[str] = Field(default_factory=lambda: ["isFraud", "fraud", "target", "label", "class"])
    drop_identifier_columns: List[str] = Field(default_factory=lambda: ["nameOrig", "nameDest"])
    leakage_risk_columns: List[str] = Field(default_factory=lambda: ["isFlaggedFraud"])
    duplicate_policy: Literal["keep_all", "drop_exact"] = "drop_exact"
    sample_max_rows: int | None = 750000


class SchemaConfig(BaseModel):
    target_col: str = "isFraud"
    positive_class_value: int = 1
    id_cols: List[str] = Field(default_factory=list)
    time_col: str | None = "step"


class SplitConfig(BaseModel):
    strategy: Literal["random_stratified", "time_based"] = "random_stratified"
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    @model_validator(mode="after")
    def validate_sum(self) -> "SplitConfig":
        if abs(self.train_size + self.val_size + self.test_size - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must sum to 1.0")
        return self


class PreprocessingConfig(BaseModel):
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    scale_numeric_for_linear: bool = True
    one_hot_min_frequency: float | None = 0.01


class CalibrationConfig(BaseModel):
    enabled: bool = True
    method: Literal["isotonic", "sigmoid"] = "isotonic"


class ThresholdConfig(BaseModel):
    default_threshold: float = 0.5
    precision_floor: float = 0.90
    recall_floor: float = 0.80
    false_positive_cost: float = 1.0
    false_negative_cost: float = 25.0


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    level: str = "INFO"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    project_name: str
    random_seed: int = 42
    paths: PathsConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    schema_cfg: SchemaConfig = Field(alias="schema")
    split: SplitConfig
    preprocessing: PreprocessingConfig
    models: Dict[str, Any]
    calibration: CalibrationConfig
    thresholding: ThresholdConfig
    api: ApiConfig
    logging: LoggingConfig

    @property
    def schema(self) -> SchemaConfig:
        return self.schema_cfg


def load_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig.model_validate(raw)
