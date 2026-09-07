from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def save_eda_plots(df, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=20, kde=True)
    plt.title("Customer Age Distribution")
    plt.tight_layout()
    plt.savefig(output / "age_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Annual_Income_(k$)"], bins=20, kde=True)
    plt.title("Annual Income Distribution")
    plt.tight_layout()
    plt.savefig(output / "income_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Spending_Score_(1-100)"], bins=20, kde=True)
    plt.title("Spending Score Distribution")
    plt.tight_layout()
    plt.savefig(output / "spending_distribution.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Annual_Income_(k$)",
        y="Spending_Score_(1-100)",
        hue="Gender",
    )
    plt.title("Income vs Spending Score")
    plt.tight_layout()
    plt.savefig(output / "income_vs_spending.png", dpi=220)
    plt.close()

def save_clustering_plots(
    df,
    labels,
    inertias,
    silhouette_scores,
    k_values,
    output_dir,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(output / "elbow_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(output / "silhouette_scores.png", dpi=220)
    plt.close()

    plot_df = df.copy()
    plot_df["Cluster"] = labels.astype(str)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="Annual_Income_(k$)",
        y="Spending_Score_(1-100)",
        hue="Cluster",
        palette="tab10",
        s=80,
    )
    plt.title("Customer Segments")
    plt.tight_layout()
    plt.savefig(output / "customer_segments.png", dpi=220)
    plt.close()

def save_pca_plot(pca_df, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        s=80,
    )
    plt.title("Customer Segments in PCA Space")
    plt.tight_layout()
    plt.savefig(output / "pca_clusters.png", dpi=220)
    plt.close()
