from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import os

# ── Rutas dentro del contenedor ──
DATA_DIR = "/opt/airflow/data"
MODEL_PATH = f"{DATA_DIR}/fraud_autoencoder.pt"
SCALER_PATH = f"{DATA_DIR}/scaler.pkl"
THRESHOLD_PATH = f"{DATA_DIR}/best_threshold.pkl"

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

# ── Tarea 1: Verificar datos ──
def verificar_datos(**context):
    csv_path = f"{DATA_DIR}/creditcard.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset no encontrado en {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset encontrado: {len(df):,} transacciones")
    print(f"   Fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    context['ti'].xcom_push(key='n_rows', value=len(df))
    return len(df)

# ── Tarea 2: Evaluar modelo actual ──
def evaluar_modelo(**context):
    df = pd.read_csv(f"{DATA_DIR}/creditcard.csv")
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    df['Amount_scaled'] = scaler.transform(df[['Amount']])
    df['Time_scaled'] = scaler.transform(df[['Time']])
    feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount']]

    X = df[feature_cols].values.astype(np.float32)
    y = df['Class'].values

    model = FraudAutoencoder(input_dim=30)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        X_reconstructed = model(X_tensor).numpy()

    errors = np.mean((X - X_reconstructed) ** 2, axis=1)
    auc = roc_auc_score(y, errors)

    print(f"📊 AUC-ROC actual: {auc:.4f}")
    context['ti'].xcom_push(key='current_auc', value=float(auc))

    if auc < 0.90:
        print(f"⚠️ AUC-ROC ({auc:.4f}) por debajo del umbral 0.90 — se re-entrenará")
    else:
        print(f"✅ Modelo en buen estado — no requiere re-entrenamiento")

    return float(auc)

# ── Tarea 3: Re-entrenar si es necesario ──
def reentrenar_modelo(**context):
    auc = context['ti'].xcom_pull(key='current_auc', task_ids='evaluar_modelo')

    if auc >= 0.90:
        print(f"✅ AUC-ROC {auc:.4f} >= 0.90 — re-entrenamiento no necesario")
        return "skipped"

    print(f"🔄 Iniciando re-entrenamiento (AUC actual: {auc:.4f})")

    df = pd.read_csv(f"{DATA_DIR}/creditcard.csv")
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount']]

    X_normal = df[df['Class'] == 0][feature_cols].values.astype(np.float32)

    model = FraudAutoencoder(input_dim=30)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_normal)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(20):
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20 completado")

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Modelo re-entrenado y guardado")
    return "retrained"

# ── Tarea 4: Registrar en log ──
def registrar_resultado(**context):
    auc = context['ti'].xcom_pull(key='current_auc', task_ids='evaluar_modelo')
    resultado = context['ti'].xcom_pull(task_ids='reentrenar_modelo')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"{timestamp} | AUC-ROC: {auc:.4f} | Acción: {resultado}\n"
    log_path = f"{DATA_DIR}/retraining_log.txt"

    with open(log_path, 'a') as f:
        f.write(log_entry)

    print(f"📝 Log registrado: {log_entry.strip()}")
    return log_entry

# ── Definición del DAG ──
default_args = {
    'owner': 'santiago',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fraud_model_monitoring',
    default_args=default_args,
    description='Monitoreo y re-entrenamiento automático del modelo de fraude',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['fraud', 'mlops', 'autoencoder'],
) as dag:

    t1 = PythonOperator(
        task_id='verificar_datos',
        python_callable=verificar_datos,
    )

    t2 = PythonOperator(
        task_id='evaluar_modelo',
        python_callable=evaluar_modelo,
    )

    t3 = PythonOperator(
        task_id='reentrenar_modelo',
        python_callable=reentrenar_modelo,
    )

    t4 = PythonOperator(
        task_id='registrar_resultado',
        python_callable=registrar_resultado,
    )

    t1 >> t2 >> t3 >> t4