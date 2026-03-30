"""
Data loading and preprocessing utilities for Insurance Fraud Detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List


def load_data(filepath: str) -> pd.DataFrame:
    """Load the insurance claims dataset."""
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame, target_col: str = 'fraud_reported') -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, List[str]]:
    """
    Preprocess the dataset: drop irrelevant columns, handle missing values,
    encode categorical variables, and prepare features and target.

    Returns:
        X: Feature DataFrame
        y: Target Series
        le_sex: LabelEncoder for sex
        feature_names: List of feature column names
    """
    # Drop irrelevant columns
    cols_to_drop = ["_c39", "policy_number", "policy_bind_date", "incident_date",
                    "incident_location", "insured_zip", "incident_city", "auto_model"]
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Handle missing values (replace '?' with mode)
    for col in df_clean.select_dtypes(include="object").columns:
        if '?' in df_clean[col].values:
            mode_val = df_clean[df_clean[col] != "?"][col].mode()[0]
            df_clean[col] = df_clean[col].replace("?", mode_val)

    # Encode target
    df_clean[target_col] = (df_clean[target_col] == "Y").astype(int)

    # Label encode insured_sex
    le_sex = LabelEncoder()
    df_clean["insured_sex"] = le_sex.fit_transform(df_clean["insured_sex"])

    # One-hot encode other categorical columns
    cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
    df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    # Ensure boolean columns are int
    for col in df_clean.columns:
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(int)

    # Separate features and target
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]

    feature_names = X.columns.tolist()

    return X, y, le_sex, feature_names


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_categorical_mappings(df: pd.DataFrame, cat_cols: List[str], feature_names: List[str]) -> dict:
    """Get mappings for categorical columns to one-hot encoded columns."""
    cat_mappings = {}
    for col in cat_cols:
        ohe_cols = [c for c in feature_names if c.startswith(col + "_")]
        cat_mappings[col] = ohe_cols
    return cat_mappings