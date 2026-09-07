import yaml
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data.download import download_dataset
from src.data.preprocessing import preprocess_dataframe
from src.features.engineering import engineer_features, MODEL_FEATURES
from src.models.train import (
    evaluate_k_values,
    select_best_k,
    train_final_model,
    save_model,
)
from src.models.evaluate import (
    create_cluster_profiles,
    create_profile_table,
    create_pca_projection,
    save_results,
)
from src.visualization.plots import (
    save_eda_plots,
    save_clustering_plots,
    save_pca_plot,
)

def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths = config["paths"]
    random_state = config["random_state"]

    print("=" * 70)
    print("CUSTOMER SEGMENTATION — K-MEANS PIPELINE")
    print("=" * 70)

    print("\n[1] Loading dataset")
    df = download_dataset(paths["raw_data"])
    print(f"Customers loaded: {len(df)}")

    print("\n[2] Preprocessing")
    df = preprocess_dataframe(
        df,
        paths["processed_data"],
    )

    print("\n[3] Feature engineering")
    df = engineer_features(df)

    print("\n[4] Exploratory analysis")
    save_eda_plots(
        df,
        paths["figures"],
    )

    X = df[MODEL_FEATURES]

    print("\n[5] Feature scaling")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[6] Searching for optimal K")

    results = evaluate_k_values(
        X_scaled,
        config["k_min"],
        config["k_max"],
        random_state=random_state,
    )

    selected_k = config.get("selected_k")

    if selected_k is None:
        selected_k = select_best_k(results)

    print(f"Selected K: {selected_k}")

    print("\n[7] Training final K-Means model")

    model, labels = train_final_model(
        X_scaled,
        selected_k,
        random_state=random_state,
    )

    results["selected"] = results["k"].eq(selected_k)

    print("\n[8] Cluster profiling")

    profile_table = create_profile_table(
        df,
        labels,
    )

    detailed_profiles = create_cluster_profiles(
        df,
        labels,
    )

    print("\nCluster summary:")
    print(profile_table.to_string(index=False))

    print("\n[9] Visualization")

    save_clustering_plots(
        df,
        labels,
        results["inertia"].tolist(),
        results["silhouette_score"].tolist(),
        results["k"].tolist(),
        paths["figures"],
    )

    pca_df, explained_variance = create_pca_projection(
        X_scaled,
        labels,
    )

    save_pca_plot(
        pca_df,
        paths["figures"],
    )

    print(
        f"PCA explained variance: "
        f"{explained_variance.sum():.2%}"
    )

    print("\n[10] Saving reports")

    save_results(
        results,
        profile_table,
        "reports",
    )

    detailed_profiles.to_csv(
        "reports/detailed_cluster_profiles.csv"
    )

    print("\n[11] Saving model")

    save_model(
        model,
        scaler,
        paths["model"],
        paths["scaler"],
    )

    print(f"Model: {paths['model']}")
    print(f"Scaler: {paths['scaler']}")

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
