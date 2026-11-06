import unittest

from src.business import retention_impact


class TestBusiness(unittest.TestCase):
    def test_simple_example(self):
        # 2 churners (annual 1200 each), model flags 3 customers incl. both churners
        r = retention_impact([100, 100, 100, 100], [1, 1, 0, 0], [1, 1, 1, 0],
                             offer_cost_rate=0.1, save_rate=0.5)
        self.assertEqual(r["churners_caught"], 2)
        self.assertEqual(r["customers_flagged"], 3)
        self.assertAlmostEqual(r["campaign_cost"], 360.0)      # 3*1200*0.1
        self.assertAlmostEqual(r["revenue_saved"], 1200.0)     # 2*1200*0.5
        self.assertAlmostEqual(r["net_benefit"], 840.0)

    def test_no_flags_means_zero_benefit(self):
        r = retention_impact([100, 200], [1, 0], [0, 0])
        self.assertEqual(r["net_benefit"], 0.0)


if __name__ == "__main__":
    unittest.main()
