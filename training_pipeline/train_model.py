"""Train the AATaaS XGBoost access-decision classifier."""

from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from training_pipeline.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from training_pipeline.data_loader import load_data
from training_pipeline.preprocessing import build_preprocessor


def train_model():
    """Train and save an XGBoost model pipeline. Returns artifacts and test split."""
    data = load_data()
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    model_bundle = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "model_name": "XGBClassifier",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model_bundle, MODEL_PATH)

    return model_bundle, X_test, y_test


if __name__ == "__main__":
    train_model()
    print(f"Model saved to {MODEL_PATH}")
