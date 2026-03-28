"""
Prepare pickle files for the Insurance Fraud Detection backend.
Loads the dataset, preprocesses it, trains the best model (XGBoost),
and exports the scaler, encoders, and model as pickle files.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

# ── 1. Load dataset ──────────────────────────────────────────────────────────
df = pd.read_csv("../insurance_claims.csv")
print(f"Dataset loaded: {df.shape}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
# Drop empty column and irrelevant columns (identifiers, dates, granular locations)
cols_to_drop = ["_c39", "policy_number", "policy_bind_date", "incident_date",
                "incident_location", "insured_zip", "incident_city", "auto_model"]
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Replace '?' with mode for each column
for col in df_clean.select_dtypes(include="object").columns:
    qm_count = (df_clean[col] == "?").sum()
    if qm_count > 0:
        mode_val = df_clean[df_clean[col] != "?"][col].mode()[0]
        df_clean[col] = df_clean[col].replace("?", mode_val)
        print(f"  '{col}': {qm_count} '?' replaced with mode '{mode_val}'")

# Encode target variable
df_clean["fraud_reported"] = (df_clean["fraud_reported"] == "Y").astype(int)

# Label encode insured_sex (binary)
le_sex = LabelEncoder()
df_clean["insured_sex"] = le_sex.fit_transform(df_clean["insured_sex"])

# Identify categorical columns for one-hot encoding
cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
print(f"\nCategorical columns for One-Hot Encoding: {cat_cols}")

# One-Hot encode remaining categorical columns
df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

# Ensure boolean columns are int
for col in df_clean.columns:
    if df_clean[col].dtype == bool:
        df_clean[col] = df_clean[col].astype(int)

# Separate features and target
X = df_clean.drop("fraud_reported", axis=1)
y = df_clean["fraud_reported"]

feature_names = X.columns.tolist()
print(f"\nFeatures ({len(feature_names)}): {feature_names}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. Train XGBoost with GridSearchCV ────────────────────────────────────────
print("\nRunning GridSearchCV for XGBoost...")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# ── 5. Save artifacts ────────────────────────────────────────────────────────
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved: scaler.pkl")

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved: model.pkl")

with open("label_encoder_sex.pkl", "wb") as f:
    pickle.dump(le_sex, f)
print("Saved: label_encoder_sex.pkl")

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("Saved: feature_names.pkl")

# Save categorical column mappings for the API
# We need to know which one-hot columns exist for each categorical variable
cat_mappings = {}
for col in cat_cols:
    ohe_cols = [c for c in feature_names if c.startswith(col + "_")]
    cat_mappings[col] = ohe_cols
with open("cat_mappings.pkl", "wb") as f:
    pickle.dump(cat_mappings, f)
print("Saved: cat_mappings.pkl")

print("\nAll artifacts saved successfully!")
