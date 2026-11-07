import pytest
from fastapi.testclient import TestClient

from credit_risk.api import main as api
from credit_risk.api.schemas import Applicant

EXAMPLE = Applicant.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def client(trained_model):
    api._state["model"] = trained_model
    with TestClient(api.app) as c:
        yield c
    api._state["model"] = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_predict(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    assert body["decision"] in {"approve", "decline"}
    assert body["model_version"] == "test"
    assert len(body["top_risk_factors"]) <= 3


def test_predict_rejects_protected_attribute(client):
    r = client.post("/predict", json={**EXAMPLE, "AGE": 30})
    assert r.status_code == 422


def test_predict_rejects_invalid_pay_status(client):
    r = client.post("/predict", json={**EXAMPLE, "PAY_1": 15})
    assert r.status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"applicants": [EXAMPLE, {**EXAMPLE, "PAY_1": -1}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_degraded_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    api._state["model"] = None
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=EXAMPLE).status_code == 503
    api._state["model"] = None
