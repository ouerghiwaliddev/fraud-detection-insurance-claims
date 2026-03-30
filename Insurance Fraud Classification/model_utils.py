"""
Model training and evaluation utilities for Insurance Fraud Detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple
import pickle


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_knn(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> KNeighborsClassifier:
    """Train KNN classifier with grid search."""
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> LogisticRegression:
    """Train Logistic Regression with grid search."""
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> DecisionTreeClassifier:
    """Train Decision Tree with grid search."""
    param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> RandomForestClassifier:
    """Train Random Forest with grid search."""
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> XGBClassifier:
    """Train XGBoost with grid search."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), param_grid, cv=cv, scoring='f1')
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['average_precision'] = average_precision_score(y_test, y_proba)

    return metrics


def save_model_artifacts(model, scaler, le_sex, feature_names, cat_mappings, output_dir: str = 'backend'):
    """Save model and preprocessing artifacts."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {
        'model.pkl': model,
        'scaler.pkl': scaler,
        'label_encoder_sex.pkl': le_sex,
        'feature_names.pkl': feature_names,
        'cat_mappings.pkl': cat_mappings
    }

    for filename, obj in artifacts.items():
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(obj, f)
        print(f"Saved: {filename}")


def load_model_artifacts(input_dir: str = 'backend'):
    """Load model and preprocessing artifacts."""
    import os

    artifacts = {}
    files = ['model.pkl', 'scaler.pkl', 'label_encoder_sex.pkl', 'feature_names.pkl', 'cat_mappings.pkl']

    for filename in files:
        with open(os.path.join(input_dir, filename), 'rb') as f:
            artifacts[filename.replace('.pkl', '')] = pickle.load(f)

    return artifacts