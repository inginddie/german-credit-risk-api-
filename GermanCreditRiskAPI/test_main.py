import pytest
from fastapi.testclient import TestClient
from main import app
from models import ClienteInput

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "api" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_predict_valid():
    payload = {
        "Age": 35,
        "Sex": 1,
        "Job": 2,
        "Housing": 1,
        "Saving_accounts": 1,
        "Checking_account": 1,
        "Credit_amount": 1500.0,
        "Duration": 12,
        "Purpose": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)  # 503 si el modelo no está cargado
    if response.status_code == 200:
        data = response.json()
        assert "risk" in data
        assert "probability_good" in data
        assert "probability_bad" in data
        assert "recommendation" in data
    else:
        assert response.json()["detail"] == "Modelo no disponible"

def test_predict_batch_valid():
    payload = [{
        "Age": 35,
        "Sex": 1,
        "Job": 2,
        "Housing": 1,
        "Saving_accounts": 1,
        "Checking_account": 1,
        "Credit_amount": 1500.0,
        "Duration": 12,
        "Purpose": 4
    } for _ in range(3)]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["total"] == 3
        assert isinstance(data["predicciones"], list)
    else:
        assert response.json()["detail"] == "Modelo no disponible"
