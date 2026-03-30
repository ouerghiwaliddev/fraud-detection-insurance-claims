"""
FastAPI backend for Insurance Fraud Detection.
Loads the pre-trained XGBoost model and scaler, exposes a /predict endpoint.
Extended with training and analysis endpoints.
"""

import pickle
import os
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import sys
sys.path.append('..')

from data_utils import load_data, preprocess_data, scale_features, get_categorical_mappings
from model_utils import (split_data, train_knn, train_logistic_regression, train_decision_tree,
                         train_random_forest, train_xgboost, evaluate_model, save_model_artifacts)
from visualization_utils import setup_plot_style

# ── Load artifacts (resolve paths relative to this file) ──────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder_sex.pkl"), "rb") as f:
    le_sex = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

with open(os.path.join(BASE_DIR, "cat_mappings.pkl"), "rb") as f:
    cat_mappings = pickle.load(f)


# ── Pydantic schema with defaults (median/mode from the dataset) ─────────────
class ClaimInput(BaseModel):
    months_as_customer: int = Field(default=200, description="Months as customer")
    age: int = Field(default=39, description="Age of the insured")
    policy_state: str = Field(default="OH", description="Policy state")
    policy_csl: str = Field(default="250/500", description="Combined single limits")
    policy_deductable: int = Field(default=1000, description="Policy deductible")
    policy_annual_premium: float = Field(default=1200.0, description="Annual premium amount")
    umbrella_limit: int = Field(default=0, description="Umbrella limit")
    insured_sex: str = Field(default="FEMALE", description="Sex: MALE or FEMALE")
    insured_education_level: str = Field(default="MD", description="Education level")
    insured_occupation: str = Field(default="exec-managerial", description="Occupation")
    insured_hobbies: str = Field(default="sleeping", description="Hobbies")
    insured_relationship: str = Field(default="husband", description="Relationship")
    capital_gains: int = Field(default=0, description="Capital gains")
    capital_loss: int = Field(default=0, description="Capital loss (negative or 0)")
    incident_type: str = Field(default="Single Vehicle Collision", description="Incident type")
    collision_type: str = Field(default="Side Collision", description="Collision type")
    incident_severity: str = Field(default="Major Damage", description="Severity")
    authorities_contacted: str = Field(default="Police", description="Authorities contacted")
    incident_state: str = Field(default="SC", description="Incident state")
    incident_hour_of_the_day: int = Field(default=12, description="Hour of the incident (0-23)")
    number_of_vehicles_involved: int = Field(default=1, description="Number of vehicles involved")
    property_damage: str = Field(default="YES", description="Property damage: YES or NO")
    bodily_injuries: int = Field(default=1, description="Number of bodily injuries")
    witnesses: int = Field(default=1, description="Number of witnesses")
    police_report_available: str = Field(default="YES", description="Police report available")
    total_claim_amount: float = Field(default=52000.0, description="Total claim amount")
    injury_claim: float = Field(default=6500.0, description="Injury claim amount")
    property_claim: float = Field(default=7500.0, description="Property claim amount")
    vehicle_claim: float = Field(default=40000.0, description="Vehicle claim amount")
    auto_make: str = Field(default="Toyota", description="Auto manufacturer")
    auto_year: int = Field(default=2005, description="Year of the vehicle")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Predict whether an insurance claim is fraudulent using a tuned XGBoost model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Insurance Fraud Detection API is running. Go to /docs for Swagger UI."}


@app.post("/predict")
def predict(claim: ClaimInput):
    """Predict fraud probability for a single insurance claim."""

    # Encode insured_sex
    sex_encoded = le_sex.transform([claim.insured_sex])[0]

    # Build numerical feature vector
    data = {
        "months_as_customer": claim.months_as_customer,
        "age": claim.age,
        "policy_deductable": claim.policy_deductable,
        "policy_annual_premium": claim.policy_annual_premium,
        "umbrella_limit": claim.umbrella_limit,
        "insured_sex": int(sex_encoded),
        "capital-gains": claim.capital_gains,
        "capital-loss": claim.capital_loss,
        "incident_hour_of_the_day": claim.incident_hour_of_the_day,
        "number_of_vehicles_involved": claim.number_of_vehicles_involved,
        "bodily_injuries": claim.bodily_injuries,
        "witnesses": claim.witnesses,
        "total_claim_amount": claim.total_claim_amount,
        "injury_claim": claim.injury_claim,
        "property_claim": claim.property_claim,
        "vehicle_claim": claim.vehicle_claim,
        "auto_year": claim.auto_year,
    }

    # Build one-hot encoded features for all categorical variables
    categorical_inputs = {
        "policy_state": claim.policy_state,
        "policy_csl": claim.policy_csl,
        "insured_education_level": claim.insured_education_level,
        "insured_occupation": claim.insured_occupation,
        "insured_hobbies": claim.insured_hobbies,
        "insured_relationship": claim.insured_relationship,
        "incident_type": claim.incident_type,
        "collision_type": claim.collision_type,
        "incident_severity": claim.incident_severity,
        "authorities_contacted": claim.authorities_contacted,
        "incident_state": claim.incident_state,
        "property_damage": claim.property_damage,
        "police_report_available": claim.police_report_available,
        "auto_make": claim.auto_make,
    }

    # Set all one-hot columns to 0, then activate the correct one
    for cat_col, ohe_cols in cat_mappings.items():
        for ohe_col in ohe_cols:
            data[ohe_col] = 0

    for cat_col, value in categorical_inputs.items():
        expected_col = f"{cat_col}_{value}"
        if expected_col in data:
            data[expected_col] = 1

    # Create DataFrame with correct column order
    df_input = pd.DataFrame([data])[feature_names]

    # Scale
    X_scaled = scaler.transform(df_input)

    # Predict
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 4),
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "risk_level": "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low",
        "input_summary": {
            "age": claim.age,
            "incident_type": claim.incident_type,
            "incident_severity": claim.incident_severity,
            "total_claim_amount": claim.total_claim_amount,
            "police_report_available": claim.police_report_available,
        },
    }


# ── Helper: preprocess a raw DataFrame (same pipeline as prepare_model.py) ───
def preprocess_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing used during training to a raw DataFrame."""
    df = df_raw.copy()

    # Drop columns not used by the model
    cols_to_drop = ["_c39", "policy_number", "policy_bind_date", "incident_date",
                    "incident_location", "insured_zip", "incident_city", "auto_model",
                    "fraud_reported"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Replace '?' with mode (computed from training data – use most common value here)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace("?", np.nan)
        if df[col].isna().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # Encode insured_sex with the saved LabelEncoder
    if "insured_sex" in df.columns:
        df["insured_sex"] = le_sex.transform(df["insured_sex"])

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure boolean columns are int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Align columns with training feature set (add missing cols as 0, drop extra)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


@app.post("/predictCSV")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with insurance claims data.
    Returns a CSV with the original data + prediction, fraud_probability, label, risk_level.
    """
    contents = await file.read()
    df_raw = pd.read_csv(io.BytesIO(contents))

    # Preprocess
    df_processed = preprocess_dataframe(df_raw)

    # Scale
    X_scaled = scaler.transform(df_processed)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Build result DataFrame
    df_result = df_raw.copy()
    df_result["prediction"] = predictions.astype(int)
    df_result["fraud_probability"] = np.round(probabilities, 4)
    df_result["label"] = np.where(predictions == 1, "Fraud", "Legitimate")
    df_result["risk_level"] = np.where(
        probabilities >= 0.7, "High",
        np.where(probabilities >= 0.4, "Medium", "Low")
    )

    # Return as downloadable CSV
    output = io.StringIO()
    df_result.to_csv(output, index=False)
    output.seek(0)

@app.post("/train-model")
def train_model(model_type: str = "xgboost", test_size: float = 0.2):
    """Train a machine learning model on the dataset."""
    try:
        # Load and preprocess data
        df = load_data("../insurance_claims.csv")
        X, y, le_sex, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Get categorical mappings
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        cat_mappings = get_categorical_mappings(df, cat_cols, feature_names)

        # Train model based on type
        if model_type == "knn":
            model = train_knn(X_train_scaled, y_train)
        elif model_type == "logistic":
            model = train_logistic_regression(X_train_scaled, y_train)
        elif model_type == "decision_tree":
            model = train_decision_tree(X_train_scaled, y_train)
        elif model_type == "random_forest":
            model = train_random_forest(X_train_scaled, y_train)
        elif model_type == "xgboost":
            model = train_xgboost(X_train_scaled, y_train)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Evaluate model
        metrics = evaluate_model(model, X_test_scaled, y_test)

        # Save artifacts
        save_model_artifacts(model, scaler, le_sex, feature_names, cat_mappings)

        return {
            "model_type": model_type,
            "metrics": metrics,
            "status": "Model trained and saved successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-metrics")
def get_model_metrics():
    """Get evaluation metrics for the current model."""
    try:
        # Load test data
        df = load_data("../insurance_claims.csv")
        X, y, le_sex, feature_names = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Evaluate current model
        metrics = evaluate_model(model, X_test_scaled, y_test)
        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-summary")
def get_data_summary():
    """Get summary statistics of the dataset."""
    try:
        df = load_data("../insurance_claims.csv")

        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numerical_stats": df.describe().to_dict(),
            "categorical_counts": {col: df[col].value_counts().to_dict()
                                 for col in df.select_dtypes(include='object').columns}
        }

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-features")
def analyze_features():
    """Perform feature analysis and return insights."""
    try:
        df = load_data("../insurance_claims.csv")
        X, y, le_sex, feature_names = preprocess_data(df)

        # Correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False).head(10).to_dict()

        # Feature importance from current model (if tree-based)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_indices = np.argsort(importance)[-10:]
            feature_importance = {feature_names[i]: float(importance[i]) for i in top_indices}

        return {
            "correlations_with_target": correlations,
            "feature_importance": feature_importance
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
