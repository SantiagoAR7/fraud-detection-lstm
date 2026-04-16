import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
from main import app

client = TestClient(app)


def test_health_endpoint():
    """API debe responder en /health"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    """El endpoint raíz debe estar activo"""
    response = client.get("/")
    assert response.status_code == 200
    assert "umbral" in response.json()


def test_predecir_estructura_respuesta():
    """El endpoint /predecir debe retornar los campos correctos"""
    payload = {
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0,
        "Amount": 100.0,
        "Time": 3600.0
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "es_fraude" in data
    assert "error_reconstruccion" in data
    assert "umbral" in data
    assert "riesgo" in data
    assert isinstance(data["es_fraude"], bool)
    assert data["riesgo"] in ["Alto", "Medio", "Bajo"]


def test_predecir_input_invalido():
    """Debe fallar si faltan campos requeridos"""
    payload = {"V1": 0.0, "Amount": 100.0}
    response = client.post("/predecir", json=payload)
    assert response.status_code == 422