"""JSON API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.models.schemas import (
    ApplicationRecord, ExplanationResponse, HealthResponse, LoanApplication,
    PredictionResponse, PredictionResult,
)
from app.services import prediction_service
from ml.explainer import enabled as explainer_enabled, explain
from ml.inference import get_engine
from ml.mlflow_logger import _enabled as mlflow_enabled

router = APIRouter(prefix="/api", tags=["api"])


def _to_record(application, prediction) -> ApplicationRecord:
    return ApplicationRecord(
        id=application.id,
        created_at=application.created_at,
        decision=prediction.prediction,
        probability_default=prediction.probability,
        risk_score=prediction.risk_score,
        risk_grade=prediction.risk_grade,
        model_version=prediction.model_version,
        input_json=application.input_json,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: LoanApplication, db: Session = Depends(get_db)):
    application, result = prediction_service.run_prediction(
        db, payload.model_dump()
    )
    return PredictionResponse(
        application_id=application.id,
        prediction=PredictionResult(**result.as_dict()),
    )


# Alias required by spec: POST /applications behaves like /predict.
@router.post("/applications", response_model=PredictionResponse)
def create_application(payload: LoanApplication, db: Session = Depends(get_db)):
    return predict(payload, db)


@router.get("/applications", response_model=list[ApplicationRecord])
def list_applications(db: Session = Depends(get_db)):
    return [_to_record(a, p) for a, p in prediction_service.list_applications(db)]


@router.get("/applications/{application_id}", response_model=ApplicationRecord)
def get_application(application_id: int, db: Session = Depends(get_db)):
    row = prediction_service.get_application(db, application_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Application not found")
    return _to_record(*row)


@router.get("/applications/{application_id}/explanation", response_model=ExplanationResponse)
def explanation(application_id: int, db: Session = Depends(get_db)):
    """LLM narrative + reason codes for a stored application.

    Reason codes are recomputed deterministically from the stored input; the LLM
    narrative is best-effort (None if the explainer is disabled/unreachable).
    """
    row = prediction_service.get_application(db, application_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Application not found")
    application, _ = row
    result = get_engine().predict(application.input_json)
    return ExplanationResponse(
        application_id=application_id,
        explanation=explain(result),
        reasons_increasing_risk=result.reasons_increasing_risk,
        reasons_lowering_risk=result.reasons_lowering_risk,
    )


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    # Model
    try:
        get_engine()
        model_loaded = True
    except Exception:
        model_loaded = False
    # Database
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "unavailable"

    status = "ok" if (model_loaded and db_status == "connected") else "degraded"
    return HealthResponse(
        status=status,
        model_version=settings.model_version,
        model_loaded=model_loaded,
        database=db_status,
        mlflow="enabled" if mlflow_enabled else "disabled",
        explainer="enabled" if explainer_enabled else "disabled",
    )
