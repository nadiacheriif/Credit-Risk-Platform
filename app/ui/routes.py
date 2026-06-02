"""Server-rendered web UI (Jinja2 + Tailwind)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.models.schemas import LoanApplication
from app.services import prediction_service
from ml.inference import get_engine
from ml.mlflow_logger import _enabled as mlflow_enabled
from ml.reference import (
    CATEGORICAL_FIELDS, CATEGORICAL_LABELS, FORM_SECTIONS,
    NUMERIC_FIELDS, example_application,
)

router = APIRouter(tags=["ui"])
templates = Jinja2Templates(directory=str(settings.templates_dir))

# Field metadata shared with every form render.
_FORM_CONTEXT = {
    "sections": FORM_SECTIONS,
    "categorical_fields": CATEGORICAL_FIELDS,
    "categorical_labels": CATEGORICAL_LABELS,
    "numeric_fields": NUMERIC_FIELDS,
}

_NUMERIC_NAMES = set(NUMERIC_FIELDS.keys())


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    ctx = {"request": request, "app_name": settings.app_name,
           "defaults": example_application(), **_FORM_CONTEXT}
    return templates.TemplateResponse("index.html", ctx)


@router.post("/predict", response_class=HTMLResponse)
async def submit(request: Request, db: Session = Depends(get_db)):
    raw = dict(await request.form())

    # Coerce numeric form strings to ints before validation.
    parsed: dict = {}
    for key, value in raw.items():
        if key in _NUMERIC_NAMES:
            try:
                parsed[key] = int(value)
            except (TypeError, ValueError):
                parsed[key] = value  # let pydantic surface the error
        else:
            parsed[key] = value

    try:
        application = LoanApplication(**parsed)
    except ValidationError as exc:
        errors = [f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()]
        ctx = {"request": request, "app_name": settings.app_name,
               "defaults": {**example_application(), **parsed},
               "errors": errors, **_FORM_CONTEXT}
        return templates.TemplateResponse("index.html", ctx, status_code=422)

    _, prediction = prediction_service.run_prediction(db, application.model_dump())

    ctx = {
        "request": request,
        "app_name": settings.app_name,
        "decision": prediction.prediction,
        "probability_default": prediction.probability,
        "probability_pct": round(prediction.probability * 100, 1),
        "credit_score": prediction.risk_score,
        "risk_grade": prediction.risk_grade,
        "model_version": prediction.model_version,
        "application_id": prediction.application_id,
    }
    return templates.TemplateResponse("result.html", ctx)


@router.get("/applications", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)):
    rows = prediction_service.list_applications(db)
    records = [
        {
            "id": a.id,
            "created_at": a.created_at,
            "decision": p.prediction,
            "probability_pct": round(p.probability * 100, 1),
            "risk_score": p.risk_score,
            "risk_grade": p.risk_grade,
            "purpose": a.input_json.get("purpose"),
            "credit_amount": a.input_json.get("credit_amount"),
        }
        for a, p in rows
    ]
    ctx = {"request": request, "app_name": settings.app_name, "records": records}
    return templates.TemplateResponse("history.html", ctx)


@router.get("/health", response_class=HTMLResponse)
def health_page(request: Request, db: Session = Depends(get_db)):
    from sqlalchemy import text

    try:
        get_engine()
        model_loaded = True
    except Exception:
        model_loaded = False
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "unavailable"

    ctx = {
        "request": request,
        "app_name": settings.app_name,
        "status": "ok" if (model_loaded and db_status == "connected") else "degraded",
        "model_version": settings.model_version,
        "model_loaded": model_loaded,
        "database": db_status,
        "mlflow": "enabled" if mlflow_enabled else "disabled",
    }
    return templates.TemplateResponse("health.html", ctx)
