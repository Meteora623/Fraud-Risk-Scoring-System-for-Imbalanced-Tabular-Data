from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int = Field(..., ge=0)
    type: str = Field(..., description="Transaction type e.g. CASH_OUT, TRANSFER")
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)
    isFlaggedFraud: int | None = Field(default=None, ge=0)
    nameOrig: str | None = None
    nameDest: str | None = None


class ScoreTransactionRequest(BaseModel):
    transaction: TransactionFeatures
    threshold_mode: Literal["balanced_f1", "high_precision", "high_recall", "cost_sensitive"] = "balanced_f1"


class ScoreTransactionResponse(BaseModel):
    model: str
    risk_score: float
    predicted_class: int
    threshold_used: float
    risk_band: str
    decision_regime: str
    decision_note: str
    review_posture: str
    recommended_action: str
    regime_metrics: dict[str, float]
    cost_context: dict[str, float]
    top_factors: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class ModelSummaryResponse(BaseModel):
    primary_model: str
    feature_importance: list[dict]


class DictResponse(BaseModel):
    payload: dict[str, Any]
