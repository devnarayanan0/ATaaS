"""Load and validate the AATaaS training dataset."""

import pandas as pd

from training_pipeline.config import DATA_PATH, TARGET_COLUMN


REQUIRED_COLUMNS = {
    "agent_role",
    "agent_autonomy_level",
    "user_role",
    "requested_action",
    "tool_requested",
    "resource_type",
    "resource_sensitivity",
    "permission_match",
    "action_risk_score",
    "prompt_injection_detected",
    "data_exfiltration_risk",
    "human_approval_required",
    "previous_failed_attempts",
    "audit_log_available",
    TARGET_COLUMN,
}


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and verify the expected columns are present."""
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data
