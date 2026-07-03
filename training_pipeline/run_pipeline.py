"""Run the full model training and evaluation pipeline."""

from training_pipeline.evaluate_model import evaluate_model
from training_pipeline.train_model import train_model


def main() -> None:
    model, X_test, y_test = train_model()
    metrics = evaluate_model(model, X_test, y_test)

    print("Training complete")
    print(f"Accuracy: {metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
