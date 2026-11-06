import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deployment"))

from app import app  # noqa: E402

GOOD = {"Tenure": 3, "MonthlyCharges": 150, "TotalCharges": 450, "SeniorCitizen": 0,
        "Contract": "Month-to-month", "PaymentMethod": "Credit Card", "PaperlessBilling": "Yes"}


class TestAPI(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_home_page_loads(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Churn Predictor", r.data)

    def test_predict_ok(self):
        r = self.client.post("/predict", json=GOOD)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.get_json()), {"churn_probability", "prediction", "risk_level"})

    def test_predict_bad_input_returns_400(self):
        r = self.client.post("/predict", json={**GOOD, "Tenure": -5})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_predict_non_json_returns_400(self):
        r = self.client.post("/predict", data="not json", content_type="text/plain")
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
