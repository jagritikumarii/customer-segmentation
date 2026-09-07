import numpy as np
import pandas as pd

FEATURES = [
    "Age",
    "Annual_Income_(k$)",
    "Spending_Score_(1-100)",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Income_per_Age"] = (
        out["Annual_Income_(k$)"] /
        out["Age"].replace(0, np.nan)
    )

    out["Income_per_Age"] = out["Income_per_Age"].fillna(
        out["Income_per_Age"].median()
    )

    return out
