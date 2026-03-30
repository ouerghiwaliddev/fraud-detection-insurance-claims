"""
Visualization utilities for Insurance Fraud Detection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from typing import Dict, Any


def setup_plot_style():
    """Setup matplotlib and seaborn style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12


def plot_target_distribution(y: np.ndarray, title: str = "Distribution of Target Variable"):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Fraud Reported')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Legitimate', 'Fraud'])
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix"):
    """Plot correlation matrix for numerical features."""
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 20, title: str = "Feature Importance"):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[-top_n:]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_precision_recall_curve(y_test: np.ndarray, y_proba: np.ndarray, title: str = "Precision-Recall Curve"):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_model_comparison(models_metrics: Dict[str, Dict[str, Any]], metric: str = 'f1_score'):
    """Plot comparison of models for a specific metric."""
    model_names = list(models_metrics.keys())
    scores = [models_metrics[name][metric] for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores)
    plt.xlabel('Models')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()