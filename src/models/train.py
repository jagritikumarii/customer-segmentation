from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def evaluate_k_values(X, k_min, k_max, random_state=42):
    inertias = []
    silhouette_scores = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )

        labels = model.fit_predict(X)

        inertias.append(model.inertia_)
        silhouette_scores.append(
            silhouette_score(X, labels)
        )

    results = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
        "silhouette_score": silhouette_scores,
    })

    return results

def select_best_k(results):
    return int(
        results.loc[
            results["silhouette_score"].idxmax(),
            "k",
        ]
    )

def train_final_model(X, k, random_state=42):
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=30,
    )

    labels = model.fit_predict(X)

    return model, labels

def save_model(model, scaler, model_path, scaler_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
