"""Train the AATaaS access-decision classifier."""

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from training_pipeline.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from training_pipeline.data_loader import load_data
from training_pipeline.preprocessing import build_preprocessor


def train_model():
    """Train and save a model pipeline. Returns the model and test split."""
    data = load_data()
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
    print(f"Model saved to {MODEL_PATH}")
