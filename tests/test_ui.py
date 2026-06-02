"""Smoke tests for the rendered web UI (catches template/Jinja errors)."""
from ml.reference import example_application


def test_form_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Loan Application" in r.text


def test_submit_renders_decision_and_reason_codes(client):
    r = client.post("/predict", data=example_application())
    assert r.status_code == 200
    assert "Credit Decision" in r.text
    assert "Why this decision" in r.text          # explainability section


def test_explain_fragment_falls_back_without_llm(client):
    app_id = client.post("/api/predict", json=example_application()).json()["application_id"]
    r = client.get(f"/explain/{app_id}")
    assert r.status_code == 200
    assert "unavailable" in r.text.lower()          # LLM disabled in tests


def test_history_and_health_pages_render(client):
    assert client.get("/applications").status_code == 200
    assert client.get("/health").status_code == 200
