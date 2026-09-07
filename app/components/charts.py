import pandas as pd
import streamlit as st

def render_customer_summary(age, income, spending):
    frame = pd.DataFrame({
        "Feature": [
            "Age",
            "Annual Income (k$)",
            "Spending Score",
        ],
        "Value": [
            age,
            income,
            spending,
        ],
    })

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
    )
