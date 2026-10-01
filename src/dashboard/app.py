"""
Streamlit dashboard: fleet fuel efficiency ranking + drill-down.

Run:
    streamlit run src/dashboard/app.py
"""
import joblib
import pandas as pd
import streamlit as st

from src.models.predict import score_fleet

st.set_page_config(page_title="Fleet Fuel Efficiency", layout="wide")
st.title("🚜 Fleet Fuel Efficiency Index")
st.caption(
    "Ranks machines by how much MORE fuel they burn than an ML model expects "
    "for their actual workload. High residual % = maintenance candidate."
)

data_path = st.sidebar.text_input("Telemetry CSV path", "data/telemetry.csv")
model_path = st.sidebar.text_input("Model path", "models_artifacts/fuel_model.joblib")

try:
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    model = joblib.load(model_path)
    df_feat, summary = score_fleet(df, model)

    st.subheader("Fleet ranking (worst offenders first)")
    st.dataframe(summary, use_container_width=True)

    st.bar_chart(summary["avg_residual_pct"])

    machine = st.selectbox("Drill into a machine", summary.index)
    machine_df = df_feat[df_feat.machine_id == machine].set_index("timestamp")

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(machine_df[["fuel_rate_gal_hr", "predicted_fuel_rate_gal_hr"]].tail(200))
        st.caption("Actual vs. predicted fuel rate (last 200 hours)")
    with col2:
        st.line_chart(machine_df[["residual_pct"]].tail(200))
        st.caption("Residual % over time — sustained upward drift = developing issue")

except FileNotFoundError:
    st.warning(
        "Data or model file not found. Run these first:\n\n"
        "```\npython -m src.simulation.generate_data --out data/telemetry.csv\n"
        "python -m src.models.train --data data/telemetry.csv --out models_artifacts/\n```"
    )
