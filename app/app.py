from pathlib import Path
import pandas as pd
import streamlit as st

from src.models.predict import load_artifacts, predict_cluster
from app.components.prediction import render_prediction
from app.components.charts import render_customer_summary

MODEL_PATH = "models/customer_segmentation_kmeans.joblib"
SCALER_PATH = "models/customer_scaler.joblib"
PROFILE_PATH = "reports/cluster_profiles.csv"

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
)

st.title("Customer Segmentation")
st.caption("K-Means powered customer analytics dashboard")

try:
    model, scaler = load_artifacts(
        MODEL_PATH,
        SCALER_PATH,
    )
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

profile = None

if Path(PROFILE_PATH).exists():
    profile = pd.read_csv(PROFILE_PATH)

st.subheader("Customer Information")

left, right = st.columns(2)

with left:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
    )

    income = st.number_input(
        "Annual Income (k$)",
        min_value=1.0,
        max_value=200.0,
        value=60.0,
        step=1.0,
    )

with right:
    spending = st.slider(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50,
    )

    st.info(
        "The model uses unsupervised learning. "
        "Cluster numbers are machine-generated labels, "
        "not predefined customer categories."
    )

if st.button(
    "Assign Customer Segment",
    type="primary",
    use_container_width=True,
):
    result = predict_cluster(
        model,
        scaler,
        age,
        income,
        spending,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        render_prediction(
            result,
            profile,
        )

    with col2:
        st.subheader("Customer summary")
        render_customer_summary(
            age,
            income,
            spending,
        )
