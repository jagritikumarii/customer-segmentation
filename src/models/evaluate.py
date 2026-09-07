from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def create_cluster_profiles(df, labels):
    profile = df.copy()
    profile["Cluster"] = labels

    numeric_columns = [
        "Age",
        "Annual_Income_(k$)",
        "Spending_Score_(1-100)",
        "Income_per_Age",
    ]

    profiles = (
        profile
        .groupby("Cluster")[numeric_columns]
        .agg(["mean", "median", "count"])
        .round(2)
    )

    return profiles

def create_profile_table(df, labels):
    profile = df.copy()
    profile["Cluster"] = labels

    result = (
        profile
        .groupby("Cluster")
        .agg(
            customers=("CustomerID", "count"),
            avg_age=("Age", "mean"),
            avg_income=("Annual_Income_(k$)", "mean"),
            avg_spending=("Spending_Score_(1-100)", "mean"),
        )
        .round(2)
        .reset_index()
    )

    result["segment_value"] = (
        result["avg_income"] *
        result["avg_spending"] / 100
    ).round(2)

    return result

def create_pca_projection(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    return pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cluster": labels,
    }), pca.explained_variance_ratio_

def save_results(results, profiles, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output / "cluster_results.csv",
        index=False,
    )

    profiles.to_csv(
        output / "cluster_profiles.csv",
        index=False,
    )
