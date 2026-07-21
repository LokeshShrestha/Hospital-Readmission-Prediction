# Hospital Readmission Prediction — MLOps Pipeline

End-to-end MLOps pipeline that predicts whether a diabetic patient will be readmitted to the hospital. Built with Airflow, MLflow, FastAPI, Streamlit, and Evidently.

![Entire Workflow](Entire%20workflow.png)

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐
│  Raw Data   │───▶│   Airflow    │───▶│    MLflow     │───▶│  FastAPI   │
│ (CSV/Redis) │    │   Pipeline   │    │ Model Registry│    │ Inference  │
└─────────────┘    └──────────────┘    └───────────────┘    └─────┬──────┘
                          │                                           │
                          ▼                                           ▼
                   ┌──────────────┐                         ┌──────────────┐
                   │ Great Exp.   │                         │   Streamlit  │
                   │ Data Val.    │                         │   Frontend   │
                   └──────────────┘                         └──────┬───────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │   SQLite DB  │
                                                            │ (Predictions)│
                                                            └──────┬───────┘
                                                                   │
                                                           ┌───────▼───────┐
                                                           │   Evidently   │
                                                           │ Drift Monitor │
                                                           └───────┬───────┘
                                                                   │
                                                           ┌───────▼───────┐
                                                           │  Airflow DAG  │
                                                           │ (Retrigger)   │
                                                           └───────────────┘
```

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Orchestration   | Apache Airflow                      |
| Data Validation | Great Expectations                  |
| Experiment      | MLflow                              |
| Model           | RandomForest (scikit-learn)         |
| Serving         | FastAPI                             |
| Frontend        | Streamlit                           |
| Monitoring      | Evidently (data drift)              |
| Cache           | Redis                               |
| Storage         | MySQL/MariaDB, SQLite               |

## Files

| File                        | Purpose                                     |
|-----------------------------|---------------------------------------------|
| `hospital_readmission.py`  | Airflow DAG (validation → ingestion → preprocessing → training → registry) |
| `fastapideploy.py`         | FastAPI REST API for model inference         |
| `frontend.py`              | Streamlit UI for predictions & history       |
| `monitoring.py`            | Evidently data drift detection + DAG trigger |
| `predictions.db`           | SQLite database of past predictions          |
| `diabetic_data.csv`        | Dataset (Diabetes 130-US hospitals)          |
| `hospital_readmission.ipynb`| Exploratory notebook                       |

## Pipeline Steps

1. **Data Validation** — Great Expectations checks (nulls, value ranges, column count)
2. **Data Ingestion** — Loads CSV & new predictions into MySQL
3. **Preprocessing** — Missing value imputation, label encoding, scaling, train/test split
4. **Training** — RandomForest with `RandomizedSearchCV` hyperparameter tuning
5. **Registry** — Logs model + metrics to MLflow, registers as `HospitalReadmissionRF`
6. **Serving** — FastAPI loads latest model from MLflow registry, exposes `/predict`
7. **Frontend** — Streamlit form → API call → result saved to SQLite
8. **Monitoring** — Evidently compares current predictions vs training data; triggers retraining DAG on drift

The DAG runs on a **monthly schedule** (`@monthly`), retraining the model on accumulated data to keep predictions accurate over time.

## Quick Start

### Prerequisites
- Python 3.9+
- Redis (port 9000)
- MySQL/MariaDB (port 3308)
- MLflow Tracking Server (port 5000)
- Airflow (with Redis + MySQL backend)

### Install
```bash
pip install -r requirements.txt
```

### Run

```bash
# 1. Start Airflow scheduler & webserver
airflow scheduler & airflow webserver

# 2. Trigger DAG
airflow dags trigger hospital_readmission

# 3. Start FastAPI
uvicorn fastapideploy:app --host 127.0.0.1 --port 8000 --reload

# 4. Start Streamlit
streamlit run frontend.py --server.port 8501

# 5. Run drift monitoring
python monitoring.py
```

### API Endpoints

| Method | Endpoint     | Description                    |
|--------|-------------|--------------------------------|
| GET    | `/`         | API health + model status      |
| POST   | `/predict`  | Predict readmission risk       |
| GET    | `/status`   | Reload model & check status    |

## Model Performance

| Metric              | Value    |
|---------------------|----------|
| Accuracy            | 0.6441   |
| Precision (weighted)| 0.6435   |
| Recall (weighted)   | 0.6441   |
| F1 Score (weighted) | 0.6390   |

## Dataset

Diabetes 130-US hospitals (1999–2008) — 100k+ inpatient encounters with ~50 features including demographics, diagnoses, lab results, and medications.

## Scheduling & Retraining

The Airflow DAG (`hospital_readmission`) is **scheduled monthly** (`@monthly`), automatically retraining the model on all accumulated data — including new predictions from the Streamlit frontend.

### Drift-Triggered Retraining

`monitoring.py` computes data drift using Evidently. If drift is detected, it POSTs to Airflow's REST API to trigger an **ad-hoc pipeline run** between scheduled cycles, enabling continuous retraining as data distribution shifts.
