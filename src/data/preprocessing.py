from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

def validate_dataframe(df: pd.DataFrame):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

def preprocess_dataframe(df: pd.DataFrame, path: str) -> pd.DataFrame:
    validate_dataframe(df)

    out = df.copy()

    out.columns = [
        c.strip().replace(" ", "_")
        for c in out.columns
    ]

    out["Gender"] = out["Gender"].astype("string").str.strip()

    numeric_columns = [
        "Age",
        "Annual_Income_(k$)",
        "Spending_Score_(1-100)",
    ]

    for column in numeric_columns:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out = out.drop_duplicates()

    out = out.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    return out
