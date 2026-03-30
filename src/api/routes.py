from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    DictResponse,
    HealthResponse,
    ModelSummaryResponse,
    ScoreTransactionRequest,
    ScoreTransactionResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    scorer = request.app.state.scorer
    model_name = scorer.bundle.model_name if scorer else None
    return HealthResponse(status="ok", model_loaded=scorer is not None, model_name=model_name)


@router.post("/score_transaction", response_model=ScoreTransactionResponse)
def score_transaction(payload: ScoreTransactionRequest, request: Request) -> ScoreTransactionResponse:
    if request.app.state.scorer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training first.")
    result = request.app.state.scorer.score_record(
        payload.transaction.model_dump(exclude_none=True),
        payload.threshold_mode,
    )
    return ScoreTransactionResponse(**result)


@router.get("/model_summary", response_model=ModelSummaryResponse)
def model_summary(request: Request) -> ModelSummaryResponse:
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training first.")
    return ModelSummaryResponse(
        primary_model=scorer.bundle.model_name,
        feature_importance=scorer.bundle.feature_importance,
    )


@router.get("/threshold_summary", response_model=DictResponse)
def threshold_summary(request: Request) -> DictResponse:
    if request.app.state.scorer is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run training first.")
    return DictResponse(payload=request.app.state.scorer.bundle.thresholds)


@router.get("/metrics_summary", response_model=DictResponse)
def metrics_summary(request: Request) -> DictResponse:
    return DictResponse(payload=request.app.state.metrics_summary)
