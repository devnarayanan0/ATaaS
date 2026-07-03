# ATaaS

AI Agent Audit Trail as a Service.

This project demonstrates a simple XGBoost risk-decision pipeline for AI agent actions.
It uses audit-event data to predict whether an action should be `Allowed`,
`Blocked`, or sent for `Needs_Human_Approval`.

## Project Structure

```text
ATaaS/
├── app.py
├── data/
│   └── agent_security_risk_scores.csv
├── models/
│   └── risk_decision_model.joblib
├── reports/
│   └── metrics.json
└── training_pipeline/
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py       # preprocessing pipeline
    ├── train_model.py         # XGBoost training pipeline
    ├── evaluate_model.py
    ├── predict.py
    └── run_pipeline.py
```

## ML Flow

1. Load the CSV dataset from `data/agent_security_risk_scores.csv`
2. Apply preprocessing for categorical and numeric features
3. Train an `XGBClassifier`
4. Save the trained model bundle to `models/risk_decision_model.joblib`
5. Use `app.py` to showcase dataset insights and live predictions

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Train the Model

```bash
.venv/bin/python -m training_pipeline.run_pipeline
```

This creates:

- `models/risk_decision_model.joblib`
- `reports/metrics.json`

## Run the Demo App

```bash
.venv/bin/streamlit run app.py
```
