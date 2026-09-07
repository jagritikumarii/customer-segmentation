# Customer Segmentation using K-Means

An end-to-end unsupervised machine learning project that segments customers into meaningful behavioral groups using **K-Means clustering**.

## Highlights

- Modular ML architecture
- Automatic dataset acquisition
- Data validation and preprocessing
- Exploratory data analysis
- Feature engineering
- StandardScaler normalization
- K-Means clustering
- Elbow method
- Silhouette analysis
- Cluster profiling
- PCA dimensionality reduction
- Multiple clustering diagnostics
- Persisted production model
- Streamlit segmentation dashboard
- Unit tests
- YAML configuration
- Reproducible pipeline

## Problem Definition

Businesses often have customers with different spending patterns, income levels, and purchasing behavior.

Instead of manually defining customer groups, this project uses unsupervised learning to discover natural customer segments.

The model groups customers based on:

- Annual income
- Spending score
- Age
- Gender
- Family-related behavior where available

The core segmentation model is **K-Means clustering**.

## Project Structure

```text
customer-segmentation-kmeans/
├── app/
│   ├── app.py
│   └── components/
│       ├── charts.py
│       └── prediction.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cluster_analysis.ipynb
│   └── 03_segmentation_experiments.ipynb
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── download.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── visualization/
│       └── plots.py
├── tests/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── .gitignore
└── LICENSE
```

## Machine Learning Workflow

```text
Raw Customer Data
       ↓
Validation
       ↓
Cleaning
       ↓
Feature Engineering
       ↓
Standardization
       ↓
K-Means Experiments
       ↓
Elbow + Silhouette Analysis
       ↓
Best K Selection
       ↓
Cluster Profiling
       ↓
PCA Visualization
       ↓
Saved Model
       ↓
Streamlit Dashboard
```

## Clustering Approach

The pipeline evaluates several values of K and calculates:

- Within-cluster sum of squares / inertia
- Silhouette score

The selected cluster count is determined from the configured search range using silhouette performance, while the elbow curve is retained as a diagnostic.

## Customer Segments

The exact cluster names are generated from the resulting cluster profiles.

Typical discovered groups may resemble:

- High-value customers
- Budget-conscious customers
- Low-engagement customers
- Young high-spenders
- Mature high-income customers

The labels are not hard-coded because clustering is unsupervised.

## Dashboard

The Streamlit application allows a user to enter:

- Age
- Annual income
- Spending score
- Gender

It then returns:

- Assigned cluster
- Cluster profile
- Distance to each cluster center
- Segment interpretation
- Visualization of the customer relative to discovered segments

## Installation

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python run_pipeline.py
```

The pipeline downloads the dataset, preprocesses it, evaluates multiple cluster counts, selects the best configuration, generates reports, profiles clusters, and saves the final model.

## Launch Dashboard

```bash
streamlit run app/app.py
```

## Run Tests

```bash
pytest
```

## Generated Reports

The pipeline generates:

- Customer distributions
- Feature correlation matrix
- Elbow curve
- Silhouette score curve
- Cluster scatter plots
- PCA cluster visualization
- Cluster profile table
- Cluster size chart

## Dataset

The project uses the **Mall Customers** dataset, a commonly used educational customer-segmentation dataset containing customer demographic and spending information.

## Limitations

K-Means assumes roughly spherical clusters and is sensitive to feature scaling and the selected number of clusters. Segments should therefore be interpreted as analytical groupings rather than absolute customer identities.

Intern ID : CITS9093
