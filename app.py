import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import time
import pandas as pd
import random

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── Arquitectura del modelo ──
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.BatchNorm1d(16), nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.BatchNorm1d(16), nn.Dropout(0.2),
            nn.Linear(16, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Cargar modelo ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = FraudAutoencoder(input_dim=30)
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "data", "fraud_autoencoder.pt"),
        map_location='cpu'
    ))
    model.eval()
    scaler = joblib.load(os.path.join(BASE_DIR, "data", "scaler.pkl"))
    threshold = joblib.load(os.path.join(BASE_DIR, "data", "best_threshold.pkl"))
    return model, scaler, threshold

model, scaler, threshold = load_model()

# ── Cargar dataset para simulación ──
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "creditcard.csv"))
    return df

df = load_data()

def predict(row):
    features = [row[f'V{i}'] for i in range(1, 29)]
    amount_scaled = scaler.transform([[row['Amount']]])[0][0]
    time_scaled = scaler.transform([[row['Time']]])[0][0]
    features.extend([amount_scaled, time_scaled])
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        X_reconstructed = model(torch.FloatTensor(X)).numpy()
    error = float(np.mean((X - X_reconstructed) ** 2))
    is_fraud = error >= threshold
    return error, is_fraud

# ── UI ──
st.title("🔍 Fraud Detection System")
st.markdown("Sistema de detección de anomalías en transacciones financieras usando Autoencoder PyTorch")
st.divider()

# ── Métricas superiores ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Modelo", "Autoencoder", "PyTorch")
with col2:
    st.metric("AUC-ROC", "0.9643", "↑ vs LSTM")
with col3:
    st.metric("Umbral", f"{threshold:.4f}", "optimizado")
with col4:
    st.metric("Dataset", "284,807", "transacciones")

st.divider()

# ── Tabs ──
tab1, tab2 = st.tabs(["🎯 Predicción manual", "⚡ Simulación en tiempo real"])

with tab1:
    st.subheader("Analizar una transacción")
    st.markdown("Selecciona una transacción real del dataset para analizarla:")

    col1, col2 = st.columns(2)
    with col1:
        tipo = st.selectbox("Tipo de transacción", ["Normal (Class=0)", "Fraude (Class=1)"])
        clase = 0 if "Normal" in tipo else 1
        muestra = df[df['Class'] == clase].sample(1).iloc[0]

    if st.button("🔍 Analizar transacción", type="primary", use_container_width=True):
        with st.spinner("Analizando..."):
            time.sleep(0.5)
            error, is_fraud = predict(muestra)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Error de reconstrucción", f"{error:.4f}")
        with col2:
            st.metric("Umbral", f"{threshold:.4f}")
        with col3:
            st.metric("Resultado real", "FRAUDE" if clase == 1 else "NORMAL")

        if is_fraud:
            st.error(f"🔴 FRAUDE DETECTADO — Error {error:.4f} supera el umbral {threshold:.4f}")
        else:
            st.success(f"🟢 TRANSACCIÓN NORMAL — Error {error:.4f} por debajo del umbral")

        st.progress(min(float(error / (threshold * 3)), 1.0))
        st.caption(f"Monto: ${muestra['Amount']:.2f} | Tiempo: {muestra['Time']:.0f}s")

with tab2:
    st.subheader("Simulación de stream en tiempo real")
    st.markdown("Simula un flujo continuo de transacciones y observa las detecciones en vivo.")

    n_transacciones = st.slider("Número de transacciones a simular", 10, 100, 30)

    if st.button("▶ Iniciar simulación", type="primary", use_container_width=True):
        resultados = []
        progress_bar = st.progress(0)
        status = st.empty()
        chart_placeholder = st.empty()

        muestra_sim = df.sample(n_transacciones)

        for i, (_, row) in enumerate(muestra_sim.iterrows()):
            error, is_fraud = predict(row)
            resultados.append({
                "Transacción": i + 1,
                "Error": round(error, 4),
                "Fraude real": "Sí" if row['Class'] == 1 else "No",
                "Detectado": "🔴 FRAUDE" if is_fraud else "🟢 Normal",
                "Monto": f"${row['Amount']:.2f}"
            })

            progress_bar.progress((i + 1) / n_transacciones)
            status.markdown(f"Procesando transacción **{i+1}/{n_transacciones}**...")

            df_resultados = pd.DataFrame(resultados)
            chart_placeholder.line_chart(df_resultados.set_index("Transacción")["Error"])
            time.sleep(0.1)

        status.markdown(f"✅ Simulación completa — {n_transacciones} transacciones procesadas")

        fraudes_detectados = sum(1 for r in resultados if "FRAUDE" in r["Detectado"])
        fraudes_reales = sum(1 for r in resultados if r["Fraude real"] == "Sí")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraudes detectados", fraudes_detectados)
        with col2:
            st.metric("Fraudes reales", fraudes_reales)
        with col3:
            precision = fraudes_detectados / max(fraudes_detectados, 1)
            st.metric("Transacciones procesadas", n_transacciones)

        st.dataframe(pd.DataFrame(resultados), use_container_width=True)