"""
app/main.py

FastAPI service for the Customer Churn Prediction project.

Run locally:
    uvicorn app.main:app --reload --port 8000

Then visit:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)

Endpoints:
    GET  /health           -> service + model status
    POST /predict           -> single customer churn prediction
    POST /predict/batch      -> multiple customers at once
"""

import glob
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import ChurnPreprocessor  # noqa: E402
from app.schemas import (  # noqa: E402
    CustomerInput, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse
)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts whether a bank customer is likely to churn, based on the "
        "methodology from Tran, Le & Nguyen (2023) — IJIKM Vol. 18."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Globals populated at startup
# ----------------------------------------------------------------------
model = None
model_name = None
preprocessor: ChurnPreprocessor | None = None


def _find_best_model_path() -> str | None:
    pattern = os.path.join(MODELS_DIR, "best_model_*.pkl")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


@app.on_event("startup")
def load_artifacts():
    global model, model_name, preprocessor

    model_path = _find_best_model_path()
    if model_path is None:
        print("WARNING: No best_model_*.pkl found in models/saved_models/. "
              "Run notebook 04 first.")
    else:
        model = joblib.load(model_path)
        model_name = os.path.basename(model_path).replace("best_model_", "").replace(".pkl", "")
        print(f"Loaded model: {model_name} from {model_path}")

    if not os.path.exists(PREPROCESSOR_PATH):
        print("WARNING: preprocessor.pkl not found. Run scripts/build_artifacts.py first.")
    else:
        preprocessor = ChurnPreprocessor.load(PREPROCESSOR_PATH)
        print(f"Loaded preprocessor with {len(preprocessor.final_feature_columns)} feature columns.")


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    return "High"


def _predict_single(customer: CustomerInput) -> PredictionResponse:
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. "
                   "Run notebook 04 (training) and scripts/build_artifacts.py first."
        )

    raw_df = pd.DataFrame([customer.dict()])
    features_df = preprocessor.transform(raw_df)

    proba = model.predict_proba(features_df)[0][1]  # probability of class 1 (churn)
    prediction = "Churn" if proba >= 0.5 else "Stay"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(float(proba), 4),
        risk_level=_risk_level(proba),
        model_used=model_name or "unknown",
    )


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None and preprocessor is not None,
        model_name=model_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    return _predict_single(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(payload: BatchPredictionRequest):
    """Predict churn for multiple customers in one request."""
    predictions = [_predict_single(c) for c in payload.customers]
    return BatchPredictionResponse(predictions=predictions)
