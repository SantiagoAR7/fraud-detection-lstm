# 🔍 Fraud Detection System

> **Sistema de detección de anomalías en transacciones financieras** — Autoencoder PyTorch end-to-end con API REST, monitoreo en tiempo real con Prometheus + Grafana, orquestación con Apache Airflow y demo interactiva con Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9643-success?style=flat-square)]()
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8.1-017CEE?style=flat-square&logo=apacheairflow)](https://airflow.apache.org/)

---

## 📌 El problema de negocio

**El fraude con tarjetas de crédito cuesta más de $30 billones anuales a nivel global.**

Este proyecto construye un sistema completo de detección de anomalías que identifica transacciones fraudulentas en tiempo real, sin necesidad de datos etiquetados para el entrenamiento. El sistema aprende el patrón de transacciones normales y marca automáticamente todo lo que se desvía de ese patrón — incluyendo fraudes nunca vistos antes.

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Raw Data      │────▶│   Autoencoder    │────▶│   FastAPI REST      │
│ (284K transacc) │     │   PyTorch        │     │   /predecir POST    │
└─────────────────┘     │   AUC-ROC 0.96   │     └──────────┬──────────┘
                        └──────────────────┘                │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Streamlit App  │     │    Grafana        │◀────│    Prometheus       │
│  Demo en vivo   │     │    Dashboard      │     │    /metrics         │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                ▲
                                │
                        ┌───────┴────────┐
                        │  Apache Airflow │
                        │  Re-entrenamiento│
                        │  @weekly        │
                        └────────────────┘
```

---

## 📊 Resultados del modelo

| Métrica | LSTM (intento 1) | Autoencoder (final) |
|---------|-----------------|---------------------|
| **AUC-ROC** | 0.54 | **0.9643** |
| F1-Score (Fraude) | 0.12 | **0.75** |
| Precision (Fraude) | 0.17 | **0.80** |
| Recall (Fraude) | 0.09 | **0.71** |

> **Decisión técnica documentada:** Se intentó primero una arquitectura LSTM con ventanas deslizantes, pero el dataset no tiene ID de tarjeta — las secuencias mezclaban transacciones de diferentes usuarios, impidiendo que la red aprendiera patrones individuales. El Autoencoder resolvió esto al aprender la distribución global de transacciones normales, logrando AUC-ROC de 0.96 vs 0.54 del LSTM. Documentar decisiones fallidas es tan valioso como documentar éxitos.

### 🔍 Cómo funciona el Autoencoder

El modelo aprende a comprimir y reconstruir transacciones normales:

```
Transacción (30 features) → Encoder → Latente (8 dims) → Decoder → Reconstrucción
```

En inferencia, las transacciones fraudulentas tienen un patrón diferente al aprendido → el error de reconstrucción es alto → se marca como anomalía.

**Umbral óptimo:** 3.9326 (optimizado por F1-Score sobre el conjunto de test)

---

## 🗂️ Estructura del proyecto

```
fraud-detection-lstm/
│
├── 📁 data/
│   ├── creditcard.csv               # Dataset (Kaggle - 284,807 transacciones)
│   ├── fraud_autoencoder.pt         # Modelo PyTorch serializado
│   ├── scaler.pkl                   # StandardScaler para Amount y Time
│   ├── best_threshold.pkl           # Umbral óptimo de detección
│   └── retraining_log.txt           # Log de re-entrenamientos automáticos
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb                 # Análisis exploratorio completo
│   └── 02_autoencoder_model.ipynb   # Entrenamiento y evaluación
│
├── 📁 api/
│   └── main.py                      # API REST con FastAPI + métricas Prometheus
│
├── 📁 airflow/
│   └── dags/
│       └── fraud_retraining_dag.py  # Pipeline de monitoreo y re-entrenamiento
│
├── 📁 monitoring/
│   └── prometheus.yml               # Configuración de scraping
│
├── app.py                           # Dashboard Streamlit
├── docker-compose.yml               # Orquestación de servicios
├── Dockerfile                       # Imagen de la API
└── requirements.txt
```

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/SantiagoAR7/fraud-detection-lstm.git
cd fraud-detection-lstm
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv venv_fraud
source venv_fraud/bin/activate
pip install -r requirements.txt
```

### 3. Descargar el dataset

Descarga [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) de Kaggle y colócalo en `data/creditcard.csv`

### 4. Entrenar el modelo

```bash
jupyter lab
# Ejecuta notebooks/01_eda.ipynb y 02_autoencoder_model.ipynb en orden
```

### 5. Levantar todos los servicios

```bash
# API de predicción
cd api && uvicorn main:app --reload

# Monitoreo (Prometheus + Grafana + Airflow)
docker compose up

# Dashboard demo
streamlit run app.py
```

---

## 🔌 Uso de la API

```bash
curl -X POST "http://127.0.0.1:8000/predecir" \
  -H "Content-Type: application/json" \
  -d '{"V1": -3.04, "V2": -3.15, "V3": 1.08, "Amount": 18.0, "Time": 406.0, ...}'
```

**Respuesta:**
```json
{
  "es_fraude": true,
  "error_reconstruccion": 4.2560,
  "umbral": 3.9326,
  "riesgo": "Alto"
}
```

---

## 📐 Decisiones técnicas

**¿Por qué Autoencoder y no clasificador supervisado?**
En fraude real, los patrones cambian constantemente. Un clasificador supervisado se vuelve obsoleto cuando aparecen nuevos tipos de fraude. El Autoencoder detecta cualquier anomalía que se desvíe del patrón normal, incluyendo fraudes nunca vistos.

**¿Por qué Apache Airflow para el re-entrenamiento?**
Airflow permite definir pipelines de ML como código, con dependencias entre tareas, reintentos automáticos y trazabilidad completa. Es el estándar de la industria para orquestación de pipelines de datos.

**¿Por qué Prometheus + Grafana y no solo logs?**
Los logs son reactivos — detectas el problema después. Las métricas son proactivas — puedes configurar alertas cuando el error de reconstrucción promedio sube, lo que puede indicar data drift antes de que el modelo falle.

---

## 🛠️ Stack tecnológico

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| Modelo | `PyTorch` | Autoencoder para detección de anomalías |
| Datos | `pandas`, `scikit-learn` | Preprocesamiento |
| API | `FastAPI`, `uvicorn` | Exposición del modelo como servicio |
| Métricas | `prometheus-fastapi-instrumentator` | Exposición de métricas |
| Monitoreo | `Prometheus`, `Grafana` | Dashboards en tiempo real |
| Orquestación | `Apache Airflow` | Pipeline de re-entrenamiento automático |
| Containerización | `Docker`, `docker-compose` | Portabilidad e infraestructura |
| Demo | `Streamlit` | Dashboard interactivo |

---

## 🗺️ Roadmap

- [x] Análisis exploratorio de datos (EDA)
- [x] Modelo LSTM — descartado por limitación del dataset (documentado)
- [x] Autoencoder PyTorch — AUC-ROC 0.9643
- [x] API REST con FastAPI + métricas Prometheus
- [x] Containerización con Docker
- [x] Pipeline de re-entrenamiento con Apache Airflow
- [x] Monitoreo en tiempo real con Prometheus + Grafana
- [x] Dashboard interactivo con Streamlit
- [ ] GitHub Actions — CI/CD automatizado
- [ ] Alertas automáticas en Grafana cuando AUC-ROC < 0.90

---

## 📁 Dataset

**Credit Card Fraud Detection** — ULB Machine Learning Group
Fuente: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transacciones · 30 features (V1-V28 son componentes PCA) · 492 fraudes (0.173%)

---

## 👤 Autor

**Santiago Atehortúa Restrepo**
Analista de Datos & Automatizaciones → ML Engineer
[GitHub](https://github.com/SantiagoAR7) · [Proyecto Churn](https://github.com/SantiagoAR7/churn-ml-pipeline)

---

*Este proyecto es parte de un portafolio de ML progresivo. Ver también: [Churn Prediction Pipeline](https://github.com/SantiagoAR7/churn-ml-pipeline)*
