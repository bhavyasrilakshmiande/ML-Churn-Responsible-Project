from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as api


client = TestClient(api.app)
VALID_PAYLOAD = {
    "tenure_months": 18,
    "support_tickets": 1,
    "monthly_spend_inr": 1299,
    "last_login_days": 2,
    "plan_type": "Pro",
}


def test_health_reports_loaded_model():
    assert api.model is not None, "Integration tests require the saved Task-04 artifact."
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_valid_prediction_uses_real_model_and_returns_probabilities():
    assert api.model is not None, "Integration tests require the saved Task-04 artifact."
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predicted_class", "class_probabilities"}
    assert body["predicted_class"] in {0, 1}
    assert set(body["class_probabilities"]) == {"0", "1"}
    assert all(0.0 <= score <= 1.0 for score in body["class_probabilities"].values())
    assert sum(body["class_probabilities"].values()) == pytest.approx(1.0)


def test_missing_required_feature_returns_validation_error():
    payload = VALID_PAYLOAD.copy()
    payload.pop("last_login_days")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any(error["loc"][-1] == "last_login_days" for error in response.json()["detail"])
    assert "input" not in response.text.lower()


@pytest.mark.parametrize(
    "changes",
    [
        {"plan_type": "Enterprise"},
        {"tenure_months": -1},
        {"support_tickets": -1},
        {"monthly_spend_inr": -1},
        {"last_login_days": -1},
        {"unexpected": "not allowed"},
    ],
)
def test_invalid_or_unsupported_features_return_validation_error(changes):
    payload = VALID_PAYLOAD | changes

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_artifact_is_reported_as_unavailable():
    missing_path = Path(__file__).with_name("artifact_that_does_not_exist.joblib")

    loaded_model, status = api.load_champion_model(missing_path)

    assert loaded_model is None
    assert status == "missing"


def test_prediction_returns_service_unavailable_when_model_is_missing(monkeypatch):
    monkeypatch.setattr(api, "model", None)

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert response.json()["detail"] == "The prediction model is unavailable."


def test_inference_failure_does_not_expose_internal_error(monkeypatch):
    class BrokenModel:
        def predict(self, features):
            raise RuntimeError("private internal diagnostic")

    monkeypatch.setattr(api, "model", BrokenModel())

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Prediction failed. Check the submitted feature values and try again."
    )
    assert "private internal diagnostic" not in response.text
