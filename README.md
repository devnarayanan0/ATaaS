# ATaaS

AI Agent Audit Trail as a Service.

This project demonstrates a simple risk-decision pipeline for AI agent actions.
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
    ├── preprocessing.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── predict.py
    └── run_pipeline.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python -m training_pipeline.run_pipeline
```

This creates:

- `models/risk_decision_model.joblib`
- `reports/metrics.json`

## Run the Demo App

```bash
streamlit run app.py
```
