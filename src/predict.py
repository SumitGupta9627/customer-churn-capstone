"""Load the saved model and score customers."""
import joblib
import pandas as pd

from .config import CATEGORICAL, CONTRACTS, MODEL_PATH, PAYMENT_METHODS
from .features import add_features

REQUIRED = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", *CATEGORICAL]


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def validate_input(record: dict) -> dict:
    """Check and clean one customer record. Raises ValueError with a clear message."""
    missing = [k for k in REQUIRED if k not in record or record[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    clean = {}
    try:
        clean["Tenure"] = int(record["Tenure"])
        clean["MonthlyCharges"] = float(record["MonthlyCharges"])
        clean["TotalCharges"] = float(record["TotalCharges"])
        clean["SeniorCitizen"] = int(record["SeniorCitizen"])
    except (TypeError, ValueError):
        raise ValueError("Tenure, MonthlyCharges, TotalCharges and SeniorCitizen must be numbers")
    if min(clean["Tenure"], clean["MonthlyCharges"], clean["TotalCharges"]) < 0:
        raise ValueError("Numeric values cannot be negative")
    if clean["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1")
    if record["Contract"] not in CONTRACTS:
        raise ValueError(f"Contract must be one of {CONTRACTS}")
    if record["PaymentMethod"] not in PAYMENT_METHODS:
        raise ValueError(f"PaymentMethod must be one of {PAYMENT_METHODS}")
    if record["PaperlessBilling"] not in ("Yes", "No"):
        raise ValueError("PaperlessBilling must be 'Yes' or 'No'")
    clean.update({k: record[k] for k in CATEGORICAL})
    return clean


def risk_band(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


def predict_one(record: dict, bundle=None) -> dict:
    bundle = bundle or load_model()
    clean = validate_input(record)
    df = add_features(pd.DataFrame([clean]))[bundle["features"]]
    prob = float(bundle["pipeline"].predict_proba(df)[0, 1])
    return {"churn_probability": round(prob, 4), "prediction": int(prob >= 0.5), "risk_level": risk_band(prob)}
