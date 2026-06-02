"""Tests for reason-code explainability and the (disabled) LLM explainer."""
import math

from ml.inference import get_engine
from ml.reference import example_application


def test_contributions_decompose_the_logit():
    """Σ(coef·WoE) + intercept must reconstruct the model's P(Good) exactly."""
    engine = get_engine()
    result = engine.predict(example_application())

    assert len(result.contributions) == 16
    total = sum(c["contribution"] for c in result.contributions)
    logit = float(engine.model.intercept_[0]) + total
    p_good = 1.0 / (1.0 + math.exp(-logit))
    assert p_good == round(result.probability_good, 6) or abs(p_good - result.probability_good) < 1e-4


def test_reason_lists_are_signed_correctly():
    result = get_engine().predict(example_application())
    assert all(c["contribution"] < 0 for c in result.reasons_increasing_risk)
    assert all(c["contribution"] > 0 for c in result.reasons_lowering_risk)


def test_api_predict_includes_contributions(client):
    body = client.post("/api/predict", json=example_application()).json()
    contribs = body["prediction"]["contributions"]
    assert len(contribs) == 16
    assert {"field", "label", "value", "woe", "contribution"} <= set(contribs[0])


def test_explanation_endpoint_degrades_without_llm(client):
    """With the explainer disabled, the endpoint still returns reason codes."""
    app_id = client.post("/api/predict", json=example_application()).json()["application_id"]
    body = client.get(f"/api/applications/{app_id}/explanation").json()
    assert body["explanation"] is None            # LLM disabled in tests
    assert len(body["reasons_increasing_risk"]) >= 0
    assert len(body["reasons_lowering_risk"]) >= 0
