"""API endpoint tests (run against an ephemeral Postgres)."""
from ml.reference import example_application


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["database"] == "connected"
    assert body["status"] == "ok"


def test_predict_returns_decision(client):
    r = client.post("/api/predict", json=example_application())
    assert r.status_code == 200
    body = r.json()
    assert body["application_id"] > 0
    pred = body["prediction"]
    assert pred["decision"] in {"Approve", "Reject", "Manual Review"}
    assert 0.0 <= pred["probability_default"] <= 1.0
    assert abs(pred["probability_default"] + pred["probability_good"] - 1.0) < 1e-6
    assert 300 <= pred["credit_score"] <= 850
    assert pred["risk_grade"] in set("ABCDEF")


def test_predict_persists_and_is_retrievable(client):
    created = client.post("/api/predict", json=example_application()).json()
    app_id = created["application_id"]

    got = client.get(f"/api/applications/{app_id}")
    assert got.status_code == 200
    rec = got.json()
    assert rec["id"] == app_id
    assert rec["decision"] == created["prediction"]["decision"]
    assert rec["input_json"]["purpose"] == example_application()["purpose"]


def test_applications_list_grows(client):
    before = len(client.get("/api/applications").json())
    client.post("/api/predict", json=example_application())
    after = len(client.get("/api/applications").json())
    assert after == before + 1


def test_get_missing_application_404(client):
    assert client.get("/api/applications/99999999").status_code == 404


def test_predict_rejects_invalid_payload(client):
    bad = example_application() | {"age": 5}  # below legal lending age
    assert client.post("/api/predict", json=bad).status_code == 422
