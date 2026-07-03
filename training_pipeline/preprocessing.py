"""Preprocessing helpers for categorical and numeric model features."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "agent_role",
    "user_role",
    "requested_action",
    "tool_requested",
    "resource_type",
]

NUMERIC_FEATURES = [
    "agent_autonomy_level",
    "resource_sensitivity",
    "permission_match",
    "action_risk_score",
    "prompt_injection_detected",
    "data_exfiltration_risk",
    "human_approval_required",
    "previous_failed_attempts",
    "audit_log_available",
]


def build_preprocessor() -> ColumnTransformer:
    """Create feature transformations for model training and inference."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
