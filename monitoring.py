import sqlite3
import pickle
import numpy as np
import pandas as pd
import os
import requests
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

pkl_dir = os.path.expanduser("~/data")

with open(os.path.join(pkl_dir, "train_X.pkl"), "rb") as f:
    X_train = pickle.load(f)

# Ensure X_train is a DataFrame so it has .columns attribute
if isinstance(X_train, np.ndarray):
    # You might need to provide column names if they aren't stored in the pickle
    # For now, we'll try to keep it as a DataFrame if possible
    X_train = pd.DataFrame(X_train)

print(f"Loaded X_train ({X_train.shape}) from {pkl_dir}/train_X.pkl")

db_path = "predictions.db"
current_data = None

if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        current_data = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()

        if not current_data.empty:
            print(f"Loaded {len(current_data)} records from predictions.db for drift detection.")
            drop_cols = ["id", "timestamp", "prediction_raw", "readmission_status"]
            current_data = current_data.drop(columns=[col for col in drop_cols if col in current_data.columns])

            # Ensure current_data columns match X_train columns
            if hasattr(X_train, 'columns'):
                current_data = current_data.reindex(columns=X_train.columns, fill_value=0)
        else:
            current_data = None
            print("predictions.db is empty.")
    except Exception as e:
        current_data = None
        print(f"Error loading from predictions.db: {e}")

if current_data is None:
    print("No prediction data available. Cannot perform drift detection.")
    exit()

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=current_data)
report.save_html("hospital_drift_report.html")

drift_result = report.as_dict()
drift_detected = drift_result["metrics"][0]["result"]["dataset_drift"]

if drift_detected:
    print("Data drift detected! Saving new data and triggering retraining DAG...")
    response = requests.post(
        "http://localhost:8080/api/v1/dags/hospital_readmission/dagRuns",
        json={"conf": {}},
        auth=("system_admin", "admin123")  
    )
    print(f"DAG trigger status: {response.status_code}")
else:
    print("No data drift detected. Skipping retraining.")
    print("Using the existing model for predictions.")