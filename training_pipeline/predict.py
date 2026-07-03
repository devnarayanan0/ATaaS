"""Prediction helper for the trained risk decision model."""

from joblib import load
import pandas as pd


class RiskDecisionPredictor:
    """Small wrapper around the saved scikit-learn model pipeline."""

    def __init__(self, model_path):
        self.model = load(model_path)

    def predict_one(self, event: dict) -> str:
        """Predict the access decision for one agent action event."""
        event_frame = pd.DataFrame([event])
        return self.model.predict(event_frame)[0]
