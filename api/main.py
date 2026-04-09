from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
import os

app = FastAPI(title="Fraud Detection API", version="1.0")

# ── Arquitectura del modelo (debe coincidir con el entrenamiento) ──
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Cargar modelo y artefactos ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = FraudAutoencoder(input_dim=30)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "data", "fraud_autoencoder.pt"),
    map_location=torch.device('cpu')
))
model.eval()

scaler = joblib.load(os.path.join(BASE_DIR, "data", "scaler.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "data", "feature_cols.pkl"))
best_threshold = joblib.load(os.path.join(BASE_DIR, "data", "best_threshold.pkl"))

# ── Schema de entrada ──
class TransactionData(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float

@app.get("/")
def root():
    return {"mensaje": "Fraud Detection API activa ✅", "umbral": best_threshold}

@app.post("/predecir")
def predecir(transaction: TransactionData):
    data = transaction.dict()

    # Escalar Amount y Time
    amount_scaled = scaler.transform([[data['Amount']]])[0][0]
    time_scaled = scaler.transform([[data['Time']]])[0][0]

    # Construir vector de features en el orden correcto
    features = [data[f'V{i}'] for i in range(1, 29)]
    features.append(amount_scaled)
    features.append(time_scaled)

    X = np.array(features, dtype=np.float32).reshape(1, -1)
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        model.eval()
        X_reconstructed = model(X_tensor).numpy()

    reconstruction_error = float(np.mean((X - X_reconstructed) ** 2))
    is_fraud = reconstruction_error >= best_threshold

    return {
        "es_fraude": bool(is_fraud),
        "error_reconstruccion": round(float(reconstruction_error), 4),
        "umbral": round(float(best_threshold), 4),
        "riesgo": "Alto" if reconstruction_error >= best_threshold * 2
                  else "Medio" if reconstruction_error >= best_threshold * 0.7
                  else "Bajo"
    }