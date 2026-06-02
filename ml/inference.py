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
from dataclasses import dataclass, asdict, field

import joblib

from app.core.config import settings
from ml.preprocess import Preprocessor
from ml.reference import field_label, humanize_value

# Score display clamp (keeps the UI gauge in a sane band)
SCORE_MIN, SCORE_MAX = 300, 850

# How many drivers to surface as reason codes per direction.
TOP_REASONS = 3


@dataclass
class InferenceResult:
    decision: str               # "Approve" | "Reject" | "Manual Review"
    probability_default: float  # P(Bad)
    probability_good: float     # P(Good)
    credit_score: int
    risk_grade: str             # A..F
    model_version: str
    # Per-feature explanation: contribution = coef * WoE to the log-odds of "Good".
    # contribution > 0 lowers risk (toward Approve); < 0 raises risk (toward default).
    contributions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def reasons_increasing_risk(self) -> list[dict]:
        """Top adverse-action drivers (push toward default)."""
        neg = [c for c in self.contributions if c["contribution"] < 0]
        return sorted(neg, key=lambda c: c["contribution"])[:TOP_REASONS]

    @property
    def reasons_lowering_risk(self) -> list[dict]:
        """Top strengths (push toward approval)."""
        pos = [c for c in self.contributions if c["contribution"] > 0]
        return sorted(pos, key=lambda c: c["contribution"], reverse=True)[:TOP_REASONS]


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

    def _contributions(self, application: dict, features) -> list[dict]:
        """Decompose the logit into per-feature contributions (coef * WoE).

        Linear scorecard: logit(Good) = intercept + Σ coef_i * woe_i, so each
        feature's signed contribution is directly interpretable. Sorted by
        magnitude (largest driver first).
        """
        coefs = self.model.coef_[0]
        names = list(self.model.feature_names_in_)  # same order as coef_
        out = []
        for name, coef in zip(names, coefs):
            woe = float(features.iloc[0][name])
            field_name = name[:-4] if name.endswith("_woe") else name
            out.append({
                "field": field_name,
                "label": field_label(field_name),
                "value": humanize_value(field_name, application.get(field_name)),
                "woe": round(woe, 4),
                "contribution": round(float(coef) * woe, 4),
            })
        out.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return out

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
            contributions=self._contributions(application, features),
        )


# Module-level singleton (artifacts loaded once at import).
_engine: CreditRiskModel | None = None


def get_engine() -> CreditRiskModel:
    global _engine
    if _engine is None:
        _engine = CreditRiskModel()
    return _engine
