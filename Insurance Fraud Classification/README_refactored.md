# Insurance Fraud Detection - Refactored

This project has been refactored to provide a clean, modular structure for insurance fraud detection analysis and API endpoints.

## Project Structure

```
Insurance Fraud Classification/
├── data_utils.py              # Data loading and preprocessing utilities
├── model_utils.py             # Model training and evaluation utilities
├── visualization_utils.py     # Plotting and visualization functions
├── main.py                    # Main script for training and analysis
├── backend/
│   ├── app.py                 # FastAPI application with prediction and analysis endpoints
│   ├── prepare_model.py       # Script to prepare model artifacts
│   └── *.pkl                  # Model artifacts (generated)
├── insurance_claims.csv       # Dataset
├── requirements.txt           # Python dependencies
└── venv/                      # Virtual environment
```

## Features

### Modular Code Structure
- **data_utils.py**: Handles data loading, preprocessing, and feature engineering
- **model_utils.py**: Provides functions for training various ML models and evaluation
- **visualization_utils.py**: Contains plotting functions for analysis
- **main.py**: Command-line interface for training models and analysis

### API Endpoints
The FastAPI application provides the following endpoints:

- `GET /`: API information
- `POST /predict`: Predict fraud probability for a single claim
- `POST /predictCSV`: Batch prediction for CSV files
- `POST /train-model`: Train a new model
- `GET /model-metrics`: Get current model evaluation metrics
- `GET /data-summary`: Get dataset summary statistics
- `POST /analyze-features`: Perform feature analysis

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model
```bash
python main.py --action train --model xgboost
```

### Data Analysis
```bash
python main.py --action analyze
```

### Model Comparison
```bash
python main.py --action compare
```

### Running the API
```bash
uvicorn backend.app:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Available Models
- KNN (K-Nearest Neighbors)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## API Examples

### Single Prediction
```python
import requests

data = {
    "months_as_customer": 200,
    "age": 39,
    "policy_state": "OH",
    "policy_csl": "250/500",
    "policy_deductable": 1000,
    # ... other fields
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### Train New Model
```python
response = requests.post("http://localhost:8000/train-model?model_type=xgboost")
print(response.json())
```

## Refactoring Summary

The original Jupyter notebook code has been refactored into:
- Modular functions for reusability
- Clean separation of concerns
- RESTful API for web integration
- Command-line interface for batch operations
- Proper error handling and logging
- Type hints for better code documentation