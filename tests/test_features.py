import pandas as pd
from src.features.engineering import engineer_features

def test_income_age_feature():
    df = pd.DataFrame({
        "CustomerID": [1],
        "Gender": ["Male"],
        "Age": [25],
        "Annual_Income_(k$)": [50],
        "Spending_Score_(1-100)": [60],
    })

    result = engineer_features(df)

    assert "Income_per_Age" in result.columns
    assert result.loc[0, "Income_per_Age"] == 2.0
