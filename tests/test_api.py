import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

# Mock los artefactos ANTES de importar main
mock_model = MagicMock()
mock_model.return_value = MagicMock()

with patch('torch.load', return_value={}), \
     patch('joblib.load', return_value=MagicMock()), \
     patch('torch.nn.Module.load_state_dict', return_value=None):
    from main import app

client = TestClient(app)


def test_health_endpoint():
    """API debe responder en /health"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predecir_input_invalido():
    """Debe fallar si faltan campos requeridos"""
    payload = {"V1": 0.0, "Amount": 100.0}
    response = client.post("/predecir", json=payload)
    assert response.status_code == 422


def test_predecir_con_mock():
    """El endpoint /predecir retorna estructura correcta con modelo mockeado"""
    import torch
    import api.main as main_module

    fake_tensor = MagicMock()
    fake_tensor.numpy.return_value = np.zeros((1, 30), dtype=np.float32)

    mock_m = MagicMock()
    mock_m.return_value = fake_tensor

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.5]]

    main_module.model = mock_m
    main_module.scaler = mock_scaler
    main_module.feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    main_module.best_threshold = 3.9326

    payload = {
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
        "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
        "V26": 0.0, "V27": 0.0, "V28": 0.0,
        "Amount": 100.0, "Time": 3600.0
    }
    response = client.post("/predecir", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "es_fraude" in data
    assert "error_reconstruccion" in data
    assert "umbral" in data
    assert "riesgo" in data
    assert isinstance(data["es_fraude"], bool)