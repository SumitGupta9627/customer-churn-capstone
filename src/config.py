"""Central configuration so paths and column names live in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_PATH = ROOT / "deployment" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"

ID_COL = "CustomerID"
TARGET = "Churn"
NUMERIC = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["Contract", "PaymentMethod", "PaperlessBilling"]
ENGINEERED = ["AvgMonthlySpend", "IsNewCustomer", "HighCharges"]
RANDOM_STATE = 42

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Bank Transfer", "Credit Card", "Electronic Check"]
