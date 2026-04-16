# 🔍 Fraud Detection System

> **Sistema de detección de anomalías en transacciones financieras** — Autoencoder PyTorch end-to-end con API REST, monitoreo en tiempo real con Prometheus + Grafana, orquestación con Apache Airflow y demo interactiva con Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9643-success?style=flat-square)]()
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-2.8.1-017CEE?style=flat-square&logo=apacheairflow)](https://airflow.apache.org/)
[![CI](https://github.com/SantiagoAR7/fraud-detection-lstm/actions/workflows/ci.yml/badge.svg)](https://github.com/SantiagoAR7/fraud-detection-lstm/actions/workflows/ci.yml)

---

## 📌 El problema de negocio

**El fraude con tarjetas de crédito cuesta más de $30 billones anuales a nivel global.**

Este proyecto construye un sistema completo de detección de anomalías que identifica transacciones fraudulentas en tiempo real, sin necesidad de datos etiquetados para el entrenamiento. El sistema aprende el patrón de transacciones normales y marca automáticamente todo lo que se desvía de ese patrón — incluyendo fraudes nunca vistos antes.

---

## 📸 Screenshots

### API REST — FastAPI + Swagger UI
![API](docs/screenshots/api.png)

### Dashboard interactivo — Streamlit
![Streamlit](docs/screenshots/streamlit.png)

### Monitoreo — Prometheus
![Prometheus](docs/screenshots/prometheus.png)

### Alertas automáticas — Grafana
![Grafana Alerts](docs/screenshots/grafana_alerts.png)

### CI/CD — GitHub Actions
![CI](docs/screenshots/github_actions.png)

---

## 🏗️ Arquitectura del sistema