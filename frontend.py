# streamlit run frontend.py --server.port 8501
import sqlite3
import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="Hospital Readmission Prediction", page_icon="🏥", layout="wide")

API_URL = "http://localhost:8000/predict"
DB_PATH = "predictions.db"

# ── SQLite Setup ──────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            encounter_id INTEGER, patient_nbr INTEGER,
            payer_code TEXT, weight TEXT, medical_specialty TEXT,
            race INTEGER, gender INTEGER, age INTEGER,
            admission_type_id INTEGER, discharge_disposition_id INTEGER, admission_source_id INTEGER,
            time_in_hospital INTEGER, num_lab_procedures INTEGER, num_procedures INTEGER,
            num_medications INTEGER, number_outpatient INTEGER, number_emergency INTEGER,
            number_inpatient INTEGER, diag_1 INTEGER, diag_2 INTEGER, diag_3 INTEGER,
            number_diagnoses INTEGER, max_glu_serum INTEGER, A1Cresult INTEGER,
            metformin INTEGER, repaglinide INTEGER, nateglinide INTEGER, chlorpropamide INTEGER,
            glimepiride INTEGER, acetohexamide INTEGER, glipizide INTEGER, glyburide INTEGER,
            tolbutamide INTEGER, pioglitazone INTEGER, rosiglitazone INTEGER, acarbose INTEGER,
            miglitol INTEGER, troglitazone INTEGER, tolazamide INTEGER, examide INTEGER,
            citoglipton INTEGER, insulin INTEGER,
            glyburide_metformin INTEGER, glipizide_metformin INTEGER,
            glimepiride_pioglitazone INTEGER, metformin_rosiglitazone INTEGER,
            metformin_pioglitazone INTEGER,
            change INTEGER, diabetesMed INTEGER,
            prediction_raw INTEGER, readmission_status TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(payload: dict, prediction_raw: int, readmission_status: str):
    conn = sqlite3.connect(DB_PATH)
    cols = list(payload.keys()) + ["prediction_raw", "readmission_status", "timestamp"]
    vals = list(payload.values()) + [prediction_raw, readmission_status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    placeholders = ", ".join(["?"] * len(vals))
    col_names = ", ".join(cols)
    conn.execute(f"INSERT INTO predictions ({col_names}) VALUES ({placeholders})", vals)
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# ── Page Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "🗃️ History", "📄 JSON Predict"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🏥 Hospital Readmission Prediction")
    st.markdown("Fill in the patient details below and click **Predict** to get the readmission risk.")

    # ── Section 1: Patient Demographics ──────────────────────────────────────
    st.subheader("Patient Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        encounter_id = st.number_input("Encounter ID", min_value=0, value=2278392)
        race = st.selectbox("Race", options=[0,1,2,3,4,5],
                            format_func=lambda x: {0:"Caucasian",1:"AfricanAmerican",2:"Hispanic",3:"Asian",4:"Other",5:"Unknown"}[x])
    with col2:
        patient_nbr = st.number_input("Patient Number", min_value=0, value=8222157)
        gender = st.selectbox("Gender", options=[0,1], format_func=lambda x: {0:"Female",1:"Male"}[x])
    with col3:
        weight = st.text_input("Weight", value="?")
        age = st.selectbox("Age Group", options=list(range(10)), format_func=lambda x: f"{x*10}-{x*10+9}")

    # ── Section 2: Admission Info ─────────────────────────────────────────────
    st.subheader("Admission Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        admission_type_id = st.selectbox("Admission Type", options=[1,2,3,4,5,6,7,8],
                                         format_func=lambda x: {1:"Emergency",2:"Urgent",3:"Elective",4:"Newborn",5:"Not Available",6:"NULL",7:"Trauma Center",8:"Not Mapped"}[x])
        payer_code = st.text_input("Payer Code", value="?")
    with col2:
        discharge_disposition_id = st.selectbox("Discharge Disposition", options=list(range(1,30)), format_func=lambda x: f"Code {x}")
        medical_specialty = st.text_input("Medical Specialty", value="?")
    with col3:
        admission_source_id = st.selectbox("Admission Source", options=list(range(1,26)), format_func=lambda x: f"Code {x}")

    # ── Section 3: Hospital Stay ──────────────────────────────────────────────
    st.subheader("Hospital Stay")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=14, value=3)
    with col2:
        num_lab_procedures = st.number_input("Lab Procedures", min_value=0, max_value=132, value=40)
    with col3:
        num_procedures = st.number_input("Procedures", min_value=0, max_value=6, value=1)
    with col4:
        num_medications = st.number_input("Medications", min_value=1, max_value=81, value=15)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=42, value=0)
    with col2:
        number_emergency = st.number_input("Emergency Visits", min_value=0, max_value=76, value=0)
    with col3:
        number_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=21, value=0)
    with col4:
        number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=7)

    # ── Section 4: Diagnoses ──────────────────────────────────────────────────
    st.subheader("Diagnoses (Encoded)")
    col1, col2, col3 = st.columns(3)
    with col1:
        diag_1 = st.number_input("Primary Diagnosis", min_value=0, max_value=999, value=250)
    with col2:
        diag_2 = st.number_input("Secondary Diagnosis", min_value=0, max_value=999, value=250)
    with col3:
        diag_3 = st.number_input("Additional Diagnosis", min_value=0, max_value=999, value=250)

    # ── Section 5: Lab Results ────────────────────────────────────────────────
    st.subheader("Lab Results")
    col1, col2 = st.columns(2)
    with col1:
        max_glu_serum = st.selectbox("Max Glucose Serum", options=[0,1,2,3],
                                      format_func=lambda x: {0:"None",1:">200",2:">300",3:"Norm"}[x])
    with col2:
        A1Cresult = st.selectbox("A1C Result", options=[0,1,2,3],
                                  format_func=lambda x: {0:"None",1:">7",2:">8",3:"Norm"}[x])

    # ── Section 6: Medications ────────────────────────────────────────────────
    st.subheader("Medications")
    med_options = {0:"No", 1:"Steady", 2:"Up", 3:"Down"}
    med_format = lambda x: med_options[x]
    meds = [
        "metformin","repaglinide","nateglinide","chlorpropamide",
        "glimepiride","acetohexamide","glipizide","glyburide",
        "tolbutamide","pioglitazone","rosiglitazone","acarbose",
        "miglitol","troglitazone","tolazamide","examide",
        "citoglipton","insulin"
    ]
    med_values = {}
    cols = st.columns(3)
    for i, med in enumerate(meds):
        with cols[i % 3]:
            med_values[med] = st.selectbox(med.capitalize(), options=[0,1,2,3], format_func=med_format, key=med)

    # ── Section 7: Combination Medications ───────────────────────────────────
    st.subheader("Combination Medications")
    combo_meds = ["glyburide_metformin","glipizide_metformin","glimepiride_pioglitazone","metformin_rosiglitazone","metformin_pioglitazone"]
    combo_labels = {
        "glyburide_metformin":"Glyburide-Metformin",
        "glipizide_metformin":"Glipizide-Metformin",
        "glimepiride_pioglitazone":"Glimepiride-Pioglitazone",
        "metformin_rosiglitazone":"Metformin-Rosiglitazone",
        "metformin_pioglitazone":"Metformin-Pioglitazone"
    }
    combo_values = {}
    cols = st.columns(3)
    for i, med in enumerate(combo_meds):
        with cols[i % 3]:
            combo_values[med] = st.selectbox(combo_labels[med], options=[0,1,2,3], format_func=med_format, key=med)

    # ── Section 8: Other Flags ────────────────────────────────────────────────
    st.subheader("Other")
    col1, col2 = st.columns(2)
    with col1:
        change = st.selectbox("Medication Change", options=[0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])
    with col2:
        diabetesMed = st.selectbox("Diabetes Medication Prescribed", options=[0,1], format_func=lambda x: {0:"No",1:"Yes"}[x])

    # ── Predict Button ────────────────────────────────────────────────────────
    st.divider()

    if st.button("🔍 Predict Readmission Risk", use_container_width=True, type="primary"):
        payload = {
            "encounter_id": encounter_id,
            "patient_nbr": patient_nbr,
            "race": race, "gender": gender, "age": age,
            "weight": weight,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "payer_code": payer_code,
            "medical_specialty": medical_specialty,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
            "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        **med_values,
        **combo_values,
        "change": change,
        "diabetesMed": diabetesMed,
    }
        with st.spinner("Running prediction..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()

                status = result.get("readmission_status", "Unknown")
                raw = result.get("prediction_raw", -1)

                # ── Save to SQLite ────────────────────────────────────────────
                save_to_db(payload, raw, status)

                if raw == 1:
                    st.error(f"⚠️ **{status}**", icon="🚨")
                else:
                    st.success(f"✅ **{status}**", icon="✔️")

                st.json(result)
                st.caption("✔️ Record saved to database.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Make sure your FastAPI server is running on `localhost:8000`.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API error: {e.response.text}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("🗃️ Prediction History")

    df = load_history()

    if df.empty:
        st.info("No predictions saved yet. Run a prediction first.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        total = len(df)
        readmitted = int((df["prediction_raw"] == 1).sum())
        not_readmitted = total - readmitted

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", total)
        col2.metric("Readmitted", readmitted)
        col3.metric("No Readmission", not_readmitted)

        st.divider()

        # ── Filter ────────────────────────────────────────────────────────────
        filter_opt = st.selectbox("Filter by result", ["All", "Readmitted", "No Readmission"])
        if filter_opt == "Readmitted":
            df = df[df["prediction_raw"] == 1]
        elif filter_opt == "No Readmission":
            df = df[df["prediction_raw"] == 0]

        # ── Table ─────────────────────────────────────────────────────────────
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────────────────
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — JSON Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("📄 JSON Prediction")
    st.markdown("Paste a JSON object matching the model's schema or upload a JSON file to run a prediction.")
    
    # ── Example JSON Structure ──
    with st.expander("See expected JSON format"):
        sample_json = {
            "encounter_id": 2278392, "patient_nbr": 8222157, "race": 0, "gender": 0, "age": 0,
            "weight": "?", "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 1,
            "time_in_hospital": 3, "payer_code": "?", "medical_specialty": "?", "num_lab_procedures": 40,
            "num_procedures": 1, "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
            "number_inpatient": 0, "diag_1": 250, "diag_2": 250, "diag_3": 250, "number_diagnoses": 7,
            "max_glu_serum": 0, "A1Cresult": 0, "metformin": 0, "repaglinide": 0, "nateglinide": 0,
            "chlorpropamide": 0, "glimepiride": 0, "acetohexamide": 0, "glipizide": 0, "glyburide": 0,
            "tolbutamide": 0, "pioglitazone": 0, "rosiglitazone": 0, "acarbose": 0, "miglitol": 0,
            "troglitazone": 0, "tolazamide": 0, "examide": 0, "citoglipton": 0, "insulin": 0,
            "glyburide_metformin": 0, "glipizide_metformin": 0, "glimepiride_pioglitazone": 0,
            "metformin_rosiglitazone": 0, "metformin_pioglitazone": 0, "change": 0, "diabetesMed": 0
        }
        st.code(json.dumps(sample_json, indent=4), language="json")

    json_input = st.text_area("Paste JSON here", height=300, placeholder='{"encounter_id": ...}')
    
    uploaded_file = st.file_uploader("Or upload JSON file", type=["json"])
    if uploaded_file is not None:
        json_input = uploaded_file.read().decode("utf-8")

    if st.button("🚀 Process JSON Prediction", use_container_width=True, type="primary"):
        if not json_input.strip():
            st.warning("Please provide JSON input.")
        else:
            try:
                payload = json.loads(json_input)
                
                # Check if it's a single record or list (handling single for simplicity)
                if isinstance(payload, list):
                    st.info(f"Detected {len(payload)} records. Processing only the first one.")
                    payload = payload[0]

                with st.spinner("Processing..."):
                    response = requests.post(API_URL, json=payload, timeout=10)
                    response.raise_for_status()
                    result = response.json()

                    status = result.get("readmission_status", "Unknown")
                    raw = result.get("prediction_raw", -1)

                    save_to_db(payload, raw, status)

                    if raw == 1:
                        st.error(f"⚠️ **{status}**", icon="🚨")
                    else:
                        st.success(f"✅ **{status}**", icon="✔️")
                    st.json(result)

            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")