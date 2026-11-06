"""Translate model output into money: a simple, transparent retention-campaign estimate."""
import numpy as np
import pandas as pd

# Assumptions (edit these to match your company)
OFFER_COST_RATE = 0.10     # retention offer costs 10% of the customer's annual revenue
SAVE_RATE = 0.30           # 30% of contacted churners accept the offer and stay


def retention_impact(monthly_charges, y_true, y_pred, offer_cost_rate=OFFER_COST_RATE,
                     save_rate=SAVE_RATE) -> dict:
    """Compare 'do nothing' with 'contact every customer the model flags'."""
    annual = np.asarray(monthly_charges, dtype=float) * 12
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    flagged = y_pred == 1
    caught = flagged & (y_true == 1)
    revenue_at_risk = float(annual[y_true == 1].sum())
    campaign_cost = float(annual[flagged].sum() * offer_cost_rate)
    revenue_saved = float(annual[caught].sum() * save_rate)
    return {
        "customers": int(len(y_true)),
        "actual_churners": int((y_true == 1).sum()),
        "customers_flagged": int(flagged.sum()),
        "churners_caught": int(caught.sum()),
        "annual_revenue_at_risk": round(revenue_at_risk, 2),
        "campaign_cost": round(campaign_cost, 2),
        "revenue_saved": round(revenue_saved, 2),
        "net_benefit": round(revenue_saved - campaign_cost, 2),
        "assumptions": {"offer_cost_rate": offer_cost_rate, "save_rate": save_rate},
    }


def churn_by_group(df: pd.DataFrame, col: str, target: str = "Churn") -> pd.DataFrame:
    out = df.groupby(col)[target].agg(customers="count", churn_rate="mean").reset_index()
    out["churn_rate"] = out["churn_rate"].round(3)
    return out
