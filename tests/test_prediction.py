import numpy as np
import pandas as pd

from src.features.engineering import engineer_features

def test_engineered_prediction_input():
    df = pd.DataFrame({
        "CustomerID": [0],
        "Gender": ["Unknown"],
        "Age": [30],
        "Annual_Income_(k$)": [60],
        "Spending_Score_(1-100)": [50],
    })

    result = engineer_features(df)

    assert result.shape[0] == 1
    assert "Income_per_Age" in result.columns
    assert np.isfinite(result["Income_per_Age"]).all()
