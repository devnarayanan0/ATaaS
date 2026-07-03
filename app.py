"""AATaaS demo app.

Run:
    streamlit run app.py

This app shows the agent security risk dataset and lets a mentor see how the
training pipeline connects to the product idea.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from training_pipeline.config import DATA_PATH, MODEL_PATH
from training_pipeline.predict import RiskDecisionPredictor


st.set_page_config(page_title="AATaaS", page_icon="A", layout="wide")

st.title("AATaaS - Agent Audit Trail as a Service")
st.caption("AI agent action risk scoring and access-decision audit demo")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


data = load_dataset()

metric_cols = st.columns(4)
metric_cols[0].metric("Audit events", len(data))
metric_cols[1].metric("Agents", data["agent_role"].nunique())
metric_cols[2].metric("Tools", data["tool_requested"].nunique())
metric_cols[3].metric("Decisions", data["access_decision"].nunique())

st.subheader("Dataset Preview")
st.dataframe(data.head(50), use_container_width=True)

st.subheader("Access Decision Distribution")
st.bar_chart(data["access_decision"].value_counts())

st.subheader("Try a Risk Decision Prediction")

if not Path(MODEL_PATH).exists():
    st.info("Train the model first with: python -m training_pipeline.run_pipeline")
else:
    predictor = RiskDecisionPredictor(MODEL_PATH)

    col1, col2, col3 = st.columns(3)
    agent_role = col1.selectbox("Agent role", sorted(data["agent_role"].unique()))
    user_role = col2.selectbox("User role", sorted(data["user_role"].unique()))
    requested_action = col3.selectbox(
        "Requested action", sorted(data["requested_action"].unique())
    )

    col4, col5, col6 = st.columns(3)
    tool_requested = col4.selectbox("Tool requested", sorted(data["tool_requested"].unique()))
    resource_type = col5.selectbox("Resource type", sorted(data["resource_type"].unique()))
    resource_sensitivity = col6.slider("Resource sensitivity", 1, 5, 3)

    col7, col8, col9 = st.columns(3)
    autonomy = col7.slider("Agent autonomy level", 1, 5, 3)
    risk_score = col8.slider("Action risk score", 0, 100, 50)
    exfiltration_risk = col9.slider("Data exfiltration risk", 0, 100, 50)

    input_event = {
        "agent_role": agent_role,
        "agent_autonomy_level": autonomy,
        "user_role": user_role,
        "requested_action": requested_action,
        "tool_requested": tool_requested,
        "resource_type": resource_type,
        "resource_sensitivity": resource_sensitivity,
        "permission_match": 1,
        "action_risk_score": risk_score,
        "prompt_injection_detected": 0,
        "data_exfiltration_risk": exfiltration_risk,
        "human_approval_required": 0,
        "previous_failed_attempts": 0,
        "audit_log_available": 1,
    }

    prediction = predictor.predict_one(input_event)
    st.success(f"Predicted access decision: {prediction}")
