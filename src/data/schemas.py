from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DatasetSchema:
    target_col: str
    id_cols: List[str]
    time_col: str | None
    dataset_name: str = "unknown"
    dataset_source: str = "unknown"
    provenance_url: str | None = None
    is_synthetic: bool = False
    dropped_identifier_cols: List[str] = field(default_factory=list)
    dropped_leakage_cols: List[str] = field(default_factory=list)
    duplicate_policy: str = "keep_all"
    duplicates_removed: int = 0
    raw_row_count: int = 0
    sampling_applied: bool = False
    sample_max_rows: int | None = None
    sampled_row_count: int = 0
