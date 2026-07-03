"""Prediction helper for the trained risk decision model."""

from joblib import load
import pandas as pd


class RiskDecisionPredictor:
    """Small wrapper around the saved XGBoost model bundle."""

    def __init__(self, model_path):
        model_bundle = load(model_path)
        self.pipeline = model_bundle["pipeline"]
        self.label_encoder = model_bundle["label_encoder"]

    def predict_one(self, event: dict) -> str:
        """Predict the access decision for one agent action event."""
        event_frame = pd.DataFrame([event])
        prediction = self.pipeline.predict(event_frame)
        return self.label_encoder.inverse_transform(prediction)[0]
