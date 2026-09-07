from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.features.engineering import engineer_features

BASE_FEATURES = [
    "Age",
    "Annual_Income_(k$)",
    "Spending_Score_(1-100)",
]

MODEL_FEATURES = BASE_FEATURES + [
    "Income_per_Age",
]

def load_artifacts(model_path, scaler_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run python run_pipeline.py first."
        )

    if not Path(scaler_path).exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run python run_pipeline.py first."
        )

    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
    )

def predict_cluster(
    model,
    scaler,
    age,
    income,
    spending,
):
    df = pd.DataFrame([{
        "CustomerID": 0,
        "Gender": "Unknown",
        "Age": age,
        "Annual_Income_(k$)": income,
        "Spending_Score_(1-100)": spending,
    }])

    df = engineer_features(df)

    X = df[MODEL_FEATURES]
    X_scaled = scaler.transform(X)

    cluster = int(model.predict(X_scaled)[0])

    distances = model.transform(X_scaled)[0]

    return {
        "cluster": cluster,
        "distances": distances,
        "features": X.iloc[0].to_dict(),
    }
