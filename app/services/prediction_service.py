"""Service layer: orchestrates inference, persistence, and MLflow logging."""
from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Application, Prediction
from ml.inference import get_engine
from ml.mlflow_logger import log_inference


def run_prediction(db: Session, application_data: dict) -> tuple[Application, Prediction]:
    """Predict, persist application + prediction, and log to MLflow.

    Returns the persisted (Application, Prediction) pair.
    """
    engine = get_engine()

    start = time.perf_counter()
    result = engine.predict(application_data)
    latency_ms = (time.perf_counter() - start) * 1000

    application = Application(input_json=application_data)
    prediction = Prediction(
        application=application,
        prediction=result.decision,
        probability=result.probability_default,
        risk_score=result.credit_score,
        risk_grade=result.risk_grade,
        model_version=result.model_version,
    )
    db.add(application)
    db.add(prediction)
    db.commit()
    db.refresh(application)
    db.refresh(prediction)

    log_inference(application_data, result.as_dict(), latency_ms)
    return application, prediction


def list_applications(db: Session, limit: int = 100) -> list[tuple[Application, Prediction]]:
    rows = db.execute(
        select(Application, Prediction)
        .join(Prediction, Prediction.application_id == Application.id)
        .order_by(Application.created_at.desc())
        .limit(limit)
    ).all()
    return [(a, p) for a, p in rows]


def get_application(db: Session, application_id: int) -> tuple[Application, Prediction] | None:
    row = db.execute(
        select(Application, Prediction)
        .join(Prediction, Prediction.application_id == Application.id)
        .where(Application.id == application_id)
    ).first()
    return (row[0], row[1]) if row else None
