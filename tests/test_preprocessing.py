import pandas as pd
from src.data.preprocessing import validate_dataframe

def test_customer_dataset_validation():
    df = pd.DataFrame({
        "CustomerID": [1],
        "Gender": ["Male"],
        "Age": [25],
        "Annual Income (k$)": [50],
        "Spending Score (1-100)": [60],
    })

    validate_dataframe(df)
