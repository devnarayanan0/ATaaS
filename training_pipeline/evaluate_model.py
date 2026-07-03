"""Evaluate the trained AATaaS access-decision model."""

import json

from sklearn.metrics import accuracy_score, classification_report

from training_pipeline.config import METRICS_PATH, REPORT_DIR


def evaluate_model(model_bundle, X_test, y_test) -> dict:
    """Evaluate a trained model bundle and save metrics as JSON."""
    pipeline = model_bundle["pipeline"]
    label_encoder = model_bundle["label_encoder"]

    predictions = pipeline.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    prediction_labels = label_encoder.inverse_transform(predictions)

    metrics = {
        "model": model_bundle["model_name"],
        "accuracy": accuracy_score(y_test_labels, prediction_labels),
        "classification_report": classification_report(
            y_test_labels,
            prediction_labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
