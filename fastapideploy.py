# uvicorn fastapideploy:app --host 127.0.0.1 --port 8000 --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import Any, Union

app = FastAPI(title="Hospital Readmission Prediction API")

mlflow.set_tracking_uri("http://localhost:5000")

class HospitalReadmissionRequest(BaseModel):
    encounter_id: Union[int, str]
    patient_nbr: Union[int, str]
    race: Any
    gender: Any
    age: Any
    weight: Any
    admission_type_id: Any
    discharge_disposition_id: Any
    admission_source_id: Any
    time_in_hospital: Any
    payer_code: Any
    medical_specialty: Any
    num_lab_procedures: Any
    num_procedures: Any
    num_medications: Any
    number_outpatient: Any
    number_emergency: Any
    number_inpatient: Any
    diag_1: Any
    diag_2: Any
    diag_3: Any
    number_diagnoses: Any
    max_glu_serum: Any
    A1Cresult: Any
    metformin: Any
    repaglinide: Any
    nateglinide: Any
    chlorpropamide: Any
    glimepiride: Any
    acetohexamide: Any
    glipizide: Any
    glyburide: Any
    tolbutamide: Any
    pioglitazone: Any
    rosiglitazone: Any
    acarbose: Any
    miglitol: Any
    troglitazone: Any
    tolazamide: Any
    examide: Any
    citoglipton: Any
    insulin: Any
    glyburide_metformin: Any
    glipizide_metformin: Any
    glimepiride_pioglitazone: Any
    metformin_rosiglitazone: Any
    metformin_pioglitazone: Any
    change: Any
    diabetesMed: Any

# Global variables for model state
model = None
model_loading_error = None

def load_mlflow_model(model_name: str, model_version: str):
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Loading model from Registry: {model_uri}")
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model, None
    except Exception as e:
        return None, str(e)

# Initialize and load the specific model version
MODEL_NAME = "HospitalReadmissionRF"
model = None
model_loading_error = None

try:
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if versions:
        MODEL_VERSION = versions[0].version 
        model, model_loading_error = load_mlflow_model(MODEL_NAME, MODEL_VERSION)
    else:
        model_loading_error = f"No versions found for model {MODEL_NAME}"
except Exception as e:
    model_loading_error = f"Could not connect to MLflow server: {str(e)}"

if model:
    print("Model loaded successfully!")
else:
    print(f"Warning: Model not available. Error: {model_loading_error}")


@app.get("/")
def root():
    return {
        "message": "Welcome to the Hospital Readmission Prediction API",
        "model_status": "loaded" if model is not None else "not_loaded",
        "error": model_loading_error
    }


@app.post("/predict")
def predict(data: HospitalReadmissionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    
    try:
        # Convert Pydantic model to dict
        input_dict = data.dict()

        # ── Mapping Logic ──
        mappings = {
            "race": {"Caucasian": 0, "AfricanAmerican": 1, "Hispanic": 2, "Asian": 3, "Other": 4, "Unknown": 5},
            "gender": {"Female": 0, "Male": 1},
            "max_glu_serum": {"None": 0, ">200": 1, ">300": 2, "Norm": 3},
            "A1Cresult": {"None": 0, ">7": 1, ">8": 2, "Norm": 3},
            "change": {"No": 0, "Ch": 1, "Yes": 1},
            "diabetesMed": {"No": 0, "Yes": 1}
        }
        
        # Medication options (applies to many fields)
        med_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
        med_fields = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", 
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", 
            "examide", "citoglipton", "insulin", "glyburide_metformin", 
            "glipizide_metformin", "glimepiride_pioglitazone", 
            "metformin_rosiglitazone", "metformin_pioglitazone"
        ]

        for k, v in input_dict.items():
            # Map category strings to ints
            if k in mappings and isinstance(v, str) and v in mappings[k]:
                input_dict[k] = mappings[k][v]
            elif k in med_fields and isinstance(v, str) and v in med_map:
                input_dict[k] = med_map[v]
            
            # Extract age from [0-10) format
            if k == "age" and isinstance(v, str) and "[" in v:
                try:
                    input_dict[k] = int(v.split("-")[0].replace("[", "")) // 10
                except: pass
            
            # Handle numeric strings for diagnosis and IDs
            if k in ["diag_1", "diag_2", "diag_3"] or "id" in k:
                if isinstance(v, str) and v != "?":
                    try: input_dict[k] = int(float(v))
                    except: input_dict[k] = 0 # Default for unknown codes
                elif v == "?":
                    input_dict[k] = 0

        # Map fields with underscores back to hyphens
        column_mapping = {
            "glyburide_metformin": "glyburide-metformin",
            "glipizide_metformin": "glipizide-metformin",
            "glimepiride_pioglitazone": "glimepiride-pioglitazone",
            "metformin_rosiglitazone": "metformin-rosiglitazone",
            "metformin_pioglitazone": "metformin-pioglitazone"
        }
        
        corrected_dict = {column_mapping.get(k, k): v for k, v in input_dict.items()}
        
        # Remove "useless" columns
        useless_cols = ['encounter_id', 'patient_nbr', 'payer_code', 'weight', 'medical_specialty']
        for col in useless_cols:
            if col in corrected_dict:
                del corrected_dict[col]
        
        # Convert to DataFrame
        df = pd.DataFrame([corrected_dict])
        
        # Ensure all columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        
        # Execute prediction
        prediction = model.predict(df)
        prediction_value = int(prediction[0])
        
        status = "Readmitted (<30 or >30 days)" if prediction_value == 1 else "No Readmission"
        
        return {
            "prediction_raw": prediction_value,
            "readmission_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/status")
def model_status():
    global model, model_loading_error
    if model is None:
        model, model_loading_error = load_mlflow_model(MODEL_NAME, MODEL_VERSION)
        if model:
            return {"status": "Model successfully reloaded"}
            
    return {
        "status": "available" if model is not None else "unavailable",
        "error": model_loading_error
    }