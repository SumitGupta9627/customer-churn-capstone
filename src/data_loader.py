"""Load and validate the customer churn dataset."""
import pandas as pd

from .config import CATEGORICAL, DATA_PATH, ID_COL, NUMERIC, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV file from disk."""
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> dict:
    """Run basic data-quality checks and return a report dictionary."""
    expected = [ID_COL, *NUMERIC, *CATEGORICAL, TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_columns": missing_cols,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COL].duplicated().sum()) if ID_COL in df else None,
        "churn_rate": float(df[TARGET].mean()) if TARGET in df else None,
        "negative_numeric_values": int((df[NUMERIC] < 0).sum().sum()) if not missing_cols else None,
    }
    report["passed"] = bool(
        not missing_cols
        and report["missing_values"] == 0
        and report["duplicate_rows"] == 0
        and report["duplicate_ids"] == 0
        and report["negative_numeric_values"] == 0
    )
    return report
