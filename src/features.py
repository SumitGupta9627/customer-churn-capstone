"""Feature engineering and the preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, ENGINEERED, NUMERIC


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven features. Works on a copy, never changes the input."""
    out = df.copy()
    out["AvgMonthlySpend"] = out["TotalCharges"] / np.maximum(out["Tenure"], 1)
    out["IsNewCustomer"] = (out["Tenure"] <= 12).astype(int)
    out["HighCharges"] = (out["MonthlyCharges"] > 100).astype(int)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    numeric_cols = NUMERIC + ENGINEERED
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
