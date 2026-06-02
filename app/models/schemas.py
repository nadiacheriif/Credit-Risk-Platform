"""Pydantic request/response schemas.

The request schema validates the RAW 20-field German-credit application:
numeric bounds come from the notebook's policy asserts; categorical fields must
be one of the documented A-codes.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from ml.reference import CATEGORICAL_FIELDS, NUMERIC_FIELDS, example_application

_N = NUMERIC_FIELDS


class LoanApplication(BaseModel):
    # --- numeric (bounds from credit-policy rules) ---
    duration: int = Field(..., ge=_N["duration"]["min"], le=_N["duration"]["max"])
    credit_amount: int = Field(..., ge=_N["credit_amount"]["min"], le=_N["credit_amount"]["max"])
    installment_rate: int = Field(..., ge=_N["installment_rate"]["min"], le=_N["installment_rate"]["max"])
    residence: int = Field(..., ge=_N["residence"]["min"], le=_N["residence"]["max"])
    age: int = Field(..., ge=_N["age"]["min"], le=_N["age"]["max"])
    credit_cards: int = Field(..., ge=_N["credit_cards"]["min"], le=_N["credit_cards"]["max"])
    dependents: int = Field(..., ge=_N["dependents"]["min"], le=_N["dependents"]["max"])

    # --- categorical (validated against documented A-codes below) ---
    account_status: str
    credit_history: str
    purpose: str
    savings: str
    employment: str
    personal_status: str
    guarantors: str
    property: str
    other_installments: str
    housing: str
    job: str
    phone: str
    foreign_worker: str

    @field_validator(*CATEGORICAL_FIELDS.keys())
    @classmethod
    def _valid_category(cls, value: str, info):
        allowed = CATEGORICAL_FIELDS[info.field_name]
        if value not in allowed:
            raise ValueError(
                f"{info.field_name}: '{value}' is not one of {sorted(allowed)}"
            )
        return value

    model_config = {"json_schema_extra": {"example": example_application()}}


class Contribution(BaseModel):
    field: str
    label: str
    value: str
    woe: float
    contribution: float   # signed: >0 lowers risk, <0 raises risk


class PredictionResult(BaseModel):
    decision: str
    probability_default: float
    probability_good: float
    credit_score: int
    risk_grade: str
    model_version: str
    contributions: list[Contribution] = []


class PredictionResponse(BaseModel):
    application_id: int
    prediction: PredictionResult


class ExplanationResponse(BaseModel):
    application_id: int
    explanation: str | None       # None when the LLM explainer is unavailable
    reasons_increasing_risk: list[Contribution]
    reasons_lowering_risk: list[Contribution]


class ApplicationRecord(BaseModel):
    id: int
    created_at: datetime
    decision: str
    probability_default: float
    risk_score: int
    risk_grade: str
    model_version: str
    input_json: dict


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    database: str
    mlflow: str
    explainer: str
