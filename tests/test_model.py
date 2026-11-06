import json
import unittest

from src.config import METRICS_PATH, MODEL_PATH
from src.predict import load_model, predict_one, validate_input

GOOD = {"Tenure": 3, "MonthlyCharges": 150, "TotalCharges": 450, "SeniorCitizen": 0,
        "Contract": "Month-to-month", "PaymentMethod": "Credit Card", "PaperlessBilling": "Yes"}
SAFE = {"Tenure": 60, "MonthlyCharges": 40, "TotalCharges": 2400, "SeniorCitizen": 0,
        "Contract": "Two year", "PaymentMethod": "Bank Transfer", "PaperlessBilling": "No"}


class TestModel(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(MODEL_PATH.exists())

    def test_metrics_meet_quality_bar(self):
        m = json.loads(METRICS_PATH.read_text())
        self.assertGreaterEqual(m["roc_auc"], 0.9)
        self.assertGreaterEqual(m["recall"], 0.8)

    def test_high_and_low_risk_customers(self):
        bundle = load_model()
        risky, safe = predict_one(GOOD, bundle), predict_one(SAFE, bundle)
        self.assertGreater(risky["churn_probability"], 0.5)
        self.assertEqual(risky["risk_level"], "High")
        self.assertLess(safe["churn_probability"], 0.3)
        self.assertEqual(safe["risk_level"], "Low")

    def test_probability_is_between_0_and_1(self):
        p = predict_one(GOOD)["churn_probability"]
        self.assertTrue(0.0 <= p <= 1.0)

    def test_invalid_inputs_are_rejected(self):
        cases = [("Tenure", -1), ("Contract", "Weekly"), ("PaperlessBilling", "Maybe"),
                 ("SeniorCitizen", 2), ("MonthlyCharges", "abc"), ("PaymentMethod", "Cash")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    validate_input({**GOOD, field: value})

    def test_missing_field_is_rejected(self):
        bad = dict(GOOD)
        del bad["Contract"]
        with self.assertRaises(ValueError):
            validate_input(bad)


if __name__ == "__main__":
    unittest.main()
