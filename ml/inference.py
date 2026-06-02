"""
Inference engine: raw application -> probability -> score -> decision.

Model output convention (from notebook):
    y = df['class'].map({1: 1, 2: 0})          # 1 = Good, 0 = Bad
    P(Good) = model.predict_proba(X)[:, 1]
    P(Default) = 1 - P(Good)

Decision engine (notebook cell 58 + scoring_config.pkl):
    best_threshold = 0.58, buffer = 0.05
    Approve        if P(Good) >= 0.63
    Reject         if P(Good) <= 0.53
    Manual Review  otherwise

Score scaling (scorecard): the notebook references `offset`/`factor` but only
persisted BASE_SCORE and PDO in scoring_config.pkl. We reconstruct the standard
points-to-double-odds scaling, anchored so odds = 1 (P=0.5) maps to BASE_SCORE:
    factor = PDO / ln(2)
    score  = BASE_SCORE + factor * ln( P(Good) / (1 - P(Good)) )
This is monotonic in creditworthiness and used only for display; it never
affects the Approve/Reject/Review decision (which is probability-based).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import joblib

from app.core.config import settings
from ml.preprocess import Preprocessor

# Score display clamp (keeps the UI gauge in a sane band)
SCORE_MIN, SCORE_MAX = 300, 850


@dataclass
class InferenceResult:
    decision: str               # "Approve" | "Reject" | "Manual Review"
    probability_default: float  # P(Bad)
    probability_good: float     # P(Good)
    credit_score: int
    risk_grade: str             # A..F
    model_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def _risk_grade(pd_default: float) -> str:
    bands = [(0.05, "A"), (0.10, "B"), (0.20, "C"),
             (0.35, "D"), (0.50, "E")]
    for threshold, grade in bands:
        if pd_default < threshold:
            return grade
    return "F"


class CreditRiskModel:
    """Loads artifacts once and serves predictions."""

    def __init__(self):
        ml = settings.ml_dir
        self.model = joblib.load(ml / "model.pkl")
        self.pre = Preprocessor(ml / "woe_bins.pkl", ml / "feature_columns.pkl")
        cfg = joblib.load(ml / "scoring_config.pkl")

        self.base_score: float = float(cfg["BASE_SCORE"])
        self.pdo: float = float(cfg["PDO"])
        self.best_threshold: float = float(cfg["best_threshold"])
        self.buffer: float = float(cfg["buffer"])
        self.factor: float = self.pdo / math.log(2)

        self.approve_threshold = self.best_threshold + self.buffer  # 0.63
        self.reject_threshold = self.best_threshold - self.buffer   # 0.53
        self.model_version = settings.model_version

    # -- scoring helpers ----------------------------------------------------
    def _score(self, p_good: float) -> int:
        p = min(max(p_good, 1e-6), 1 - 1e-6)
        raw = self.base_score + self.factor * math.log(p / (1 - p))
        return int(round(min(max(raw, SCORE_MIN), SCORE_MAX)))

    def _decision(self, p_good: float) -> str:
        if p_good >= self.approve_threshold:
            return "Approve"
        if p_good <= self.reject_threshold:
            return "Reject"
        return "Manual Review"

    # -- public API ---------------------------------------------------------
    def predict(self, application: dict) -> InferenceResult:
        features = self.pre.transform(application)
        p_good = float(self.model.predict_proba(features)[0, 1])
        p_default = 1.0 - p_good
        return InferenceResult(
            decision=self._decision(p_good),
            probability_default=round(p_default, 6),
            probability_good=round(p_good, 6),
            credit_score=self._score(p_good),
            risk_grade=_risk_grade(p_default),
            model_version=self.model_version,
        )


# Module-level singleton (artifacts loaded once at import).
_engine: CreditRiskModel | None = None


def get_engine() -> CreditRiskModel:
    global _engine
    if _engine is None:
        _engine = CreditRiskModel()
    return _engine
