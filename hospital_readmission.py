from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import redis
import pickle
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from datetime import datetime
import great_expectations as gx
from sklearn.metrics import precision_score, recall_score, f1_score
import sqlite3

def ingestion():
    redis_conn = redis.Redis(host="127.0.0.1", port=9000)
    csv_df = pickle.loads(redis_conn.get("raw_df"))
    
    # Load from predictions.db (Streamlit generate DB)
    workspace_dir = "d:/Assignments/Sem4/MLOPS/Final" # Using absolute path for reliability in Airflow
    db_path = os.path.join(workspace_dir, "predictions.db")
    
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            pred_df = pd.read_sql_query("SELECT * FROM predictions", conn)
            conn.close()
            
            if not pred_df.empty:
                print(f"Loaded {len(pred_df)} records from predictions.db")
                # Drop technical columns
                tech_cols = ['id', 'timestamp', 'readmission_status']
                pred_df = pred_df.drop(columns=[c for c in tech_cols if c in pred_df.columns])
                
                # Align column names (underscores to hyphens for meds)
                column_mapping = {
                    "glyburide_metformin": "glyburide-metformin",
                    "glipizide_metformin": "glipizide-metformin",
                    "glimepiride_pioglitazone": "glimepiride-pioglitazone",
                    "metformin_rosiglitazone": "metformin-rosiglitazone",
                    "metformin_pioglitazone": "metformin-pioglitazone"
                }
                pred_df = pred_df.rename(columns=column_mapping)
                
                # Map prediction_raw to readmitted format expected by preprocessing
                if "prediction_raw" in pred_df.columns:
                    pred_df = pred_df.rename(columns={"prediction_raw": "readmitted"})
                    pred_df['readmitted'] = pred_df['readmitted'].apply(lambda x: ">30" if x == 1 else "NO")
                
                # Ensure pred_df has all columns from csv_df to avoid issues
                # Fill missing columns with '?' for object types or 0 for numeric if needed
                # But they should match due to previous updates
                
                # Merge (Concatenate)
                df = pd.concat([csv_df, pred_df], ignore_index=True)
                print(f"Total merged records: {len(df)}")
            else:
                df = csv_df
        except Exception as e:
            print(f"Error reading predictions.db: {e}")
            df = csv_df
    else:
        print(f"predictions.db not found at {db_path}")
        df = csv_df

    password = quote_plus("Sunway@123")
    engine = create_engine(f"mysql+pymysql://mariadbuser:{password}@localhost:3308/Hospital_db")
    df.to_sql('Hospital_data', engine, if_exists='replace', index=False)
    
def data_validation():
    df = pd.read_csv('~/airflow/diabetic_data.csv')
    redis_conn = redis.Redis(host="127.0.0.1", port=9000)
    
    context = gx.get_context()
    suite_name = "hospital_expectation_suite"
    context.suites.add(gx.ExpectationSuite(name=suite_name))
    datasource = context.data_sources.add_pandas(name="hospital_pandas_datasource")
    data_asset = datasource.add_dataframe_asset(name="hospital_data")
    
    batch_definition = data_asset.add_batch_definition_whole_dataframe("hospital_batch_definition")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    validator = context.get_validator(
        batch=batch,
        expectation_suite_name=suite_name
    )
    
    validator.expect_column_values_to_not_be_null(column="encounter_id")
    validator.expect_column_values_to_be_in_set(column="readmitted", value_set=["<30", ">30", "NO"])
    
    validator.expect_column_values_to_be_in_set(
        column="gender", 
        value_set=["Female", "Male", "Unknown/Invalid"]
    )
    validator.expect_column_values_to_be_between(
        column="time_in_hospital", 
        min_value=1, 
        max_value=14
    )
    validator.expect_column_values_to_be_between(
        column="num_lab_procedures", 
        min_value=1, 
        max_value=132
    )
    validator.expect_column_values_to_be_between(
        column="num_medications", 
        min_value=1, 
        max_value=81
    )
    validator.expect_column_values_to_not_be_null(column="patient_nbr")
    validator.expect_column_values_to_not_be_null(column="diag_1")
    
    validator.expect_table_column_count_to_equal(value=50)
    
    validation_result = validator.validate()
    
    print("Data Validation success:", validation_result.success)
    if not validation_result.success:
        raise ValueError("Data validation failed based on defined expectations.")
    redis_conn.set("raw_df", pickle.dumps(df))


def preprocessing():
    redis_conn = redis.Redis(host="127.0.0.1", port=9000)
    password = quote_plus("Sunway@123")
    engine = create_engine(f"mysql+pymysql://mariadbuser:{password}@localhost:3308/Hospital_db")
    df = pd.read_sql_table("Hospital_data", engine)

    df.replace('?', np.nan, inplace=True)

    df.drop(columns=['encounter_id', 'patient_nbr', 'payer_code','weight','medical_specialty'], inplace=True)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' or x == '>30' else 0)
    df['readmitted'].value_counts()

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['readmitted', 'readmitted_binary']]

    print(f"Categorical columns to encode: {categorical_cols}")

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    y = df['readmitted']
    X = df.drop('readmitted', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"\nTraining target distribution:")
    print(y_train.value_counts())
    print(f"\nTest target distribution:")
    print(y_test.value_counts())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed")
    print(f"Scaled training data shape: {X_train_scaled.shape}")
    print(f"Scaled test data shape: {X_test_scaled.shape}")
    
    redis_conn.set("train_X", pickle.dumps(X_train_scaled))
    redis_conn.set("train_y", pickle.dumps(y_train))
    redis_conn.set("test_X", pickle.dumps(X_test_scaled))
    redis_conn.set("test_y", pickle.dumps(y_test))

    # Save to .pkl files for monitoring
    pkl_dir = os.path.expanduser("~/data")
    os.makedirs(pkl_dir, exist_ok=True)

    with open(os.path.join(pkl_dir, "train_X.pkl"), "wb") as f:
        pickle.dump(X_train_scaled, f)
    with open(os.path.join(pkl_dir, "train_y.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(pkl_dir, "test_X.pkl"), "wb") as f:
        pickle.dump(X_test_scaled, f)
    with open(os.path.join(pkl_dir, "test_y.pkl"), "wb") as f:
        pickle.dump(y_test, f)
    print(f"Saved train/test splits to {pkl_dir}")

def model_training():
    
    redis_conn = redis.Redis(host="127.0.0.1", port=9000)
    
    X_train = pickle.loads(redis_conn.get("train_X"))
    y_train = pickle.loads(redis_conn.get("train_y"))
    
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    print("Starting Hyperparameter Tuning...")
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=5, 
        cv=2, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    best_rf = random_search.best_estimator_
    print(f"Best parameters found: {random_search.best_params_}")
    
    redis_conn.set("rf_model", pickle.dumps(best_rf))

def model_registry():
    redis_conn = redis.Redis(host="127.0.0.1", port=9000)
    X_train = pickle.loads(redis_conn.get("train_X"))
    y_train = pickle.loads(redis_conn.get("train_y"))
    X_test = pickle.loads(redis_conn.get("test_X"))
    y_test = pickle.loads(redis_conn.get("test_y"))
    model = pickle.loads(redis_conn.get("rf_model"))

    mlflow.set_experiment("Hospital_Readmission_Pipeline")
    
    mlflow.sklearn.autolog()
    with mlflow.start_run():
    

        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="random_forest_model",
            registered_model_name="HospitalReadmissionRF"
        )
        
        redis_conn.set("hospital_rf_model", pickle.dumps(model))
        redis_conn.set("hospital_model_accuracy", accuracy)


with DAG(
    dag_id="hospital_readmission",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
) as dag:
    t_val = PythonOperator(task_id="DataValidation", python_callable=data_validation)
    t1 = PythonOperator(task_id="DataIngestion", python_callable=ingestion)
    t2 = PythonOperator(task_id="DataPreprocessing", python_callable=preprocessing)
    t4 = PythonOperator(task_id="ModelTraining", python_callable=model_training)
    t5 = PythonOperator(task_id="ModelDeployment", python_callable=model_registry)

    t_val >> t1 >> t2 >> t4 >> t5