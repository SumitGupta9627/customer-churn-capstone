import unittest

import pandas as pd

from src.data_loader import load_data, validate_data
from src.features import add_features


class TestData(unittest.TestCase):
    def test_data_loads_and_passes_quality_checks(self):
        report = validate_data(load_data())
        self.assertEqual(report["rows"], 500)
        self.assertTrue(report["passed"])

    def test_validate_detects_missing_values(self):
        df = load_data()
        df.loc[0, "Tenure"] = None
        self.assertFalse(validate_data(df)["passed"])

    def test_add_features_keeps_input_and_handles_zero_tenure(self):
        df = load_data().head(3).copy()
        df.loc[0, "Tenure"] = 0
        before = df.copy()
        out = add_features(df)
        pd.testing.assert_frame_equal(df, before)
        self.assertTrue({"AvgMonthlySpend", "IsNewCustomer", "HighCharges"} <= set(out.columns))
        self.assertTrue(out["AvgMonthlySpend"].notna().all())
        self.assertEqual(out.loc[0, "IsNewCustomer"], 1)


if __name__ == "__main__":
    unittest.main()
