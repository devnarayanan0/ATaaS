"""Project paths and training defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "agent_security_risk_scores.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "risk_decision_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET_COLUMN = "access_decision"
TEST_SIZE = 0.2
RANDOM_STATE = 42
